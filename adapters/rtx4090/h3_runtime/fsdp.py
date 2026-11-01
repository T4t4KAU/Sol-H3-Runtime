"""Load a mixed-dtype DiT with configurable FSDP, retaining official SP and AdaLN paths."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import ContextParallelConfig, MiniMaxH3Transformer3DModel
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard

from . import adaln, fusion_install, ulysses
from .cp_plan import MINIMAX_H3_CP_PLAN, assert_no_attention_mask
from .lora import fuse_lora, load_lora_branches


def _construct(config):
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel.from_config(config).to(dtype=torch.bfloat16)
        for name in model._keep_in_fp32_modules:
            if hasattr(model, name):
                model.get_submodule(name).float()
    return model.eval().requires_grad_(False)


def _precompute(source, video_times, audio_times, *, include_audio_condition=False):
    """Run official precompute with only one projection resident on CUDA at a time."""
    device = torch.device("cuda", torch.cuda.current_device())
    source.time_proj.to(device)
    source.time_embedder.to(device)
    handles = []

    def onload(module, _args):
        module.to(device)

    def offload(module, _args, _result):
        module.cpu()

    for block in source.transformer_blocks:
        handles.append(block.adaln_proj.register_forward_pre_hook(onload))
        handles.append(block.adaln_proj.register_forward_hook(offload))
    try:
        stats = adaln.precompute(source, video_times, audio_times, execution_device=device,
                                include_audio_condition=include_audio_condition)
    finally:
        for handle in handles:
            handle.remove()
        source.cpu()
    print(f"AdaLN: {stats}", flush=True)
    gc.collect()
    torch.cuda.empty_cache()


def _enable_parallelism(model, engine=None):
    model.set_attention_backend("native")
    degree = engine.ulysses_degree if engine is not None else dist.get_world_size()
    group = None
    if degree > 1:
        assert_no_attention_mask(model)
        mesh = (engine._get_parallel_mesh("cp") if engine is not None else
                init_device_mesh("cuda", (1, degree), mesh_dim_names=("ring", "ulysses")))
        model.enable_parallelism(
            config=ContextParallelConfig(ulysses_degree=degree, ring_degree=1, ulysses_anything=True, mesh=mesh),
            cp_plan=MINIMAX_H3_CP_PLAN,
        )
        # enable_parallelism propagates to every processor, including the unsharded refiner.
        for block in model.token_refiner.refiner_blocks:
            block.attn.processor._parallel_config = None
        group = mesh["ulysses"].get_group()
    ulysses.reset_row_counts()
    if engine is None:
        fusion_install.install(model, lora=False)
        if degree > 1:
            ulysses.install(model, group=group, attention_fn=None, lora=False)
    else:
        engine._install_transformer(model, group)


def _branch_layout(model):
    return [(name, module.r["default"], module.lora_alpha["default"], module.scaling["default"])
            for name, module in model.named_modules() if hasattr(module, "lora_A")]


def _restore_branches(model, branches):
    """Recreate the official PEFT wrappers on meta; DCP loads their actual weights."""
    if not branches:
        return
    from peft import LoraConfig
    from peft.tuners.lora import Linear

    for name, rank, alpha, scaling in branches:
        base = model.get_submodule(name)
        with torch.device("meta"):
            wrapper = Linear(base, adapter_name="default",
                             config=LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0),
                             r=rank, lora_alpha=alpha, lora_dropout=0).to(dtype=base.weight.dtype)
        wrapper.scaling["default"] = scaling
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, wrapper.eval().requires_grad_(False))


def _shard(model, degree, *, mesh=None):
    if degree == 1:
        # DCP assigns broadcast tensors into a meta target. Materializing here would
        # keep a second full CUDA model alive until load_state_dict finishes.
        return
    if mesh is None:
        mesh = init_device_mesh("cuda", (degree,), mesh_dim_names=("fsdp",))
    policy = MixedPrecisionPolicy(param_dtype=None, output_dtype=None, cast_forward_inputs=False)
    options = dict(mesh=mesh, reshard_after_forward=True, mp_policy=policy)
    for block in model.transformer_blocks:
        fully_shard(block, **options)
    for block in model.token_refiner.refiner_blocks:
        fully_shard(block, **options)
    for name in ("proj_in", "audio_proj_in", "context_embedder", "time_embedder",
                 "token_refiner.final_norm", "norm_out", "proj_out", "audio_proj_out"):
        fully_shard(model.get_submodule(name), **options)
    fully_shard(model, **options)
    # Allocate local shards only, never a full CUDA transformer.
    model.to_empty(device=torch.device("cuda", torch.cuda.current_device()))


def _layout(model):
    return [(name, tuple(p.shape), p.dtype) for name, p in model.named_parameters()]


def _check_loaded(model, layout, *, sharded):
    if model.training or _layout(model) != layout:
        raise RuntimeError("Loaded DiT dtype/shape/eval contract differs from the CPU source")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad or parameter.is_meta:
            raise RuntimeError(f"Parameter is not materialized and frozen: {name}")
        if sharded and (not isinstance(parameter, DTensor) or parameter.placements != (Shard(0),)
                        or parameter.to_local().is_meta):
            raise RuntimeError(f"Parameter was not materialized as a frozen FSDP shard: {name}")
    for name, buffer in model.named_buffers():
        if buffer.is_meta:
            raise RuntimeError(f"Unmaterialized buffer: {name}")


@torch.no_grad()
def _load_sharded(config, source, cpu_group, video_times, audio_times, *, engine=None):
    rank = dist.get_rank()
    degree = (engine.runtime.fsdp_degree or 1) if engine is not None else dist.get_world_size()
    precompute = engine is None or engine.runtime.adaln_precompute
    component = "transformer_ref" if engine is not None and engine.task == "ref2va" else "transformer"
    if rank == 0 and precompute:
        _precompute(source, video_times, audio_times, include_audio_condition=component == "transformer_ref")
    metadata = [None]
    if rank == 0:
        metadata[0] = dict(layout=_layout(source), branches=_branch_layout(source), tables=[
            (tuple(block.adaln_proj.table.shape), block.adaln_proj.table.dtype)
            for block in source.transformer_blocks
        ] if precompute else [])
    dist.broadcast_object_list(metadata, 0, group=cpu_group)
    model = _construct(config)
    _restore_branches(model, metadata[0]["branches"])
    if precompute:
        cursor = adaln._StepCursor(torch.device("cuda", torch.cuda.current_device()))
        tables = metadata[0]["tables"]
        if len(tables) != len(model.transformer_blocks):
            raise RuntimeError("AdaLN table count does not match the transformer")
        for block, (shape, dtype) in zip(model.transformer_blocks, tables):
            block.adaln_proj = adaln.PrecomputedModulation(torch.empty(shape, dtype=dtype, device="meta"), cursor)
        model._h3opt_adaln_cursor = cursor
    if _layout(model) != metadata[0]["layout"]:
        raise RuntimeError("Source and target parameter layouts differ")
    if engine is None:
        _enable_parallelism(model)
    else:
        _enable_parallelism(model, engine)
    model.eval().requires_grad_(False)
    mesh = engine._get_parallel_mesh("fsdp") if engine is not None and degree > 1 else None
    _shard(model, degree, mesh=mesh)
    # DCP mutates its input mapping: pass a copy, retaining the independent CPU source.
    state = source.state_dict() if rank == 0 else {}
    result = set_model_state_dict(model, dict(state), options=StateDictOptions(
        full_state_dict=True, broadcast_from_rank0=True, strict=True, ignore_frozen_params=False,
    ))
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Incomplete checkpoint load: {result}")

    # These nonpersistent buffers are intentionally absent from state_dict.
    canonical = MiniMaxH3RotaryPosEmbed(model.config.rope_freq_dim, model.config.rope_theta)
    if rank == 0 and not torch.equal(source.rope.inv_freq, canonical.inv_freq):
        raise RuntimeError("CPU source RoPE differs from canonical FP32 frequencies")
    model.rope.inv_freq = canonical.inv_freq.to(torch.cuda.current_device())
    if model.rope.inv_freq.dtype != torch.float32 or not torch.equal(model.rope.inv_freq.cpu(), canonical.inv_freq):
        raise RuntimeError("RoPE reconstruction failed")
    if precompute:
        for i, block in enumerate(model.transformer_blocks):
            if block.adaln_proj.table.is_meta:
                block.adaln_proj.table = torch.empty_like(block.adaln_proj.table, device=torch.cuda.current_device())
            if rank == 0:
                block.adaln_proj.table.copy_(source.transformer_blocks[i].adaln_proj.table)
            dist.broadcast(block.adaln_proj.table, 0)
        adaln.enable_adaln_precompute(model, verbose=rank == 0, component_name=component)
        # Keep the cursor hook without replacing parameter modules after FSDP.
        model._h3opt_adaln_wanted = False
    _check_loaded(model, metadata[0]["layout"], sharded=degree > 1)
    if degree > 1 and engine is not None and engine.runtime.evidence_dir:
        from .distributed_loading import audit_shards
        audit_shards(model, source, cpu_group=cpu_group,
                     directory=Path(engine.runtime.evidence_dir)/'weights')
    return model


@torch.no_grad()
def load_transformer(model_path: Path, adapter_path: Path | None, cpu_group, video_times, audio_times, *, engine=None):
    """Rank zero owns the full CPU source; all ranks receive their own CUDA shards."""
    source = None
    task = engine.task if engine is not None else "t2v"
    lora_mode = engine.lora_mode if engine is not None else "merged"
    component = "transformer_ref" if task == "ref2va" else "transformer"
    if dist.get_rank() == 0:
        source = MiniMaxH3Transformer3DModel.from_pretrained(
            str(model_path / component), torch_dtype=torch.bfloat16,
            local_files_only=True, low_cpu_mem_usage=True,
        ).eval().requires_grad_(False)
        alpha = 8 if task == "ref2va" else 64
        if adapter_path is None:
            report = {"adapter": None, "original_checkpoint": True}
        elif lora_mode == "merged":
            report = fuse_lora(source, adapter_path, alpha=alpha, execution_device="cuda")
        else:
            report = load_lora_branches(source, adapter_path, alpha=alpha)
        print(f"LoRA: {report}", flush=True)
    config = json.loads((model_path / component / "config.json").read_text())
    if engine is None:
        model = _load_sharded(config, source, cpu_group, video_times, audio_times)
    else:
        model = _load_sharded(config, source, cpu_group, video_times, audio_times, engine=engine)
    del source
    gc.collect()
    torch.cuda.empty_cache()
    return model
