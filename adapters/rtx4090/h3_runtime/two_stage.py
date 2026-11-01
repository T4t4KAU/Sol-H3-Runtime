"""Staged FSDP residency around the released H3-to-LTX bridge and LTX sampler."""
from __future__ import annotations

import gc
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import weakref

import torch
import torch.distributed as dist

from .distributed_loading import load_cpu_source
from .evidence import ObservedEuler, PredictionCapture, cpu
from .ltx_sp import NCCLAttentionManager, require_divisible_video
from .upstream import runtime


def _release():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _broadcast(value, engine):
    payload = [cpu(value) if engine.rank == 0 else None]
    dist.broadcast_object_list(payload, 0, group=engine.cpu_group)
    return tuple(t.to(engine.device) for t in payload[0])


def dense_ops(attention_backend='flash'):
    from ltx_core.model.transformer.attention import AttentionFunction, MaskedAttentionFunction
    from ltx_core.model.transformer.transformer import TransformerOpsConfig
    attention = {'flash': AttentionFunction.SDPA_FLASH,
                 'efficient': AttentionFunction.SDPA_EFFICIENT}[attention_backend]
    return TransformerOpsConfig.from_functions(attention=attention,
                                               masked_attention=MaskedAttentionFunction.SDPA_EFFICIENT)


def transformer_builder(paths, base, device):
    """The upstream loader owns key mapping, mixed dtype handling and LoRA fusion."""
    lora = base.LoraPathStrengthAndSDOps(str(paths['refiner_lora']), .8, base.LTXV_LORA_COMFY_RENAMING_MAP)
    stage = base.DiffusionStage.from_checkpoint(str(paths['transformer']), torch.bfloat16, device,
        loras=(lora,), quantization=None, registry=base.ModelRegistry(cache_weights=False, cache_models=False),
        compilation_config=None, alloc_trim_strategy=base.AllocatorTrimStrategy.DEFER,
        offload_mode=base.OffloadMode.NONE)
    return stage._transformer_builder


@torch.no_grad()
def load_transformer(paths, base, engine, *, evidence_dir=None,
                     attention_backend='flash', linear_mode='native'):
    from ltx_core.model.transformer.model import X0Model
    from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
    from ltx_core.multigpu.transformer.sequence_parallel import (
        SequenceParallelModelWrapper, create_video_self_attention_module_ops)

    builder = transformer_builder(paths, base, engine.device)
    source = None
    if engine.rank == 0:
        from ltx_core.loader.fuse_loras import FuseRule
        original_rule = builder.fuse_rule
        merged = 0
        def retain_on_cpu(key, weight, deltas, state):
            nonlocal merged
            # The native builder retains fused weights on the fusion device even
            # for a CPU build. Keep its exact math, but return each completed
            # tensor to the independent CPU source before fusing the next one.
            result = original_rule(key, weight, deltas, state)
            merged += 1
            if merged % 64 == 0:
                print(f'LTX LoRA CPU source: {merged} weights merged', flush=True)
            return {name:value.cpu() for name,value in result.items()}
        source = builder.with_fuse_rule(FuseRule(original_rule.aggregation_dtype, retain_on_cpu)).build(
            device=torch.device('cpu'), dtype=torch.bfloat16)
    metadata = builder.model_metadata()
    with torch.device('meta'):
        target = LTXModelConfigurator.from_metadata(metadata, ops=dense_ops(attention_backend))
    if len(target.transformer_blocks) != 48:
        raise ValueError('Expected the released 48-layer LTX joint transformer')
    manager = NCCLAttentionManager(engine._get_parallel_mesh('cp')['ulysses'].get_group())
    target = create_video_self_attention_module_ops(manager).mutator(target)
    layout = None
    if linear_mode == 'full-sequence':
        from .linear_layout import LinearSequenceLayout
        layout = LinearSequenceLayout(manager.rank, manager.world_size)
        layout.install(target)
    elif linear_mode != 'native':
        raise ValueError('Unknown Linear execution mode: ' + linear_mode)
    units = list(target.transformer_blocks)
    units += [m for name,m in target.named_children() if name != 'transformer_blocks'
              and any(True for _ in m.parameters())]
    target = load_cpu_source(target, source, mesh=engine._get_parallel_mesh('fsdp'),
        cpu_group=engine.cpu_group, units=units,
        audit_dir=None if evidence_dir is None else Path(evidence_dir)/'weights')
    del source, builder
    _release()
    wrapper = SequenceParallelModelWrapper(target, manager)
    wrapper.register_forward_pre_hook(require_divisible_video, with_kwargs=True)
    if layout is not None:
        wrapper.register_forward_pre_hook(layout.prepare, with_kwargs=True)
    return X0Model(wrapper).eval().requires_grad_(False), manager


@torch.no_grad()
def refine(engine, paths, payload, output, *, seed, evidence_dir=None,
           attention_backend='flash', linear_mode='native', collect_raw=False):
    """All ranks participate in FSDP/SP denoising; rank zero owns bridge and decode."""
    official = runtime(paths['sana_root'], 'stage2')
    helpers = runtime(paths['sana_root'], 'stage2_ops.models')
    adapter_api = runtime(paths['sana_root'], 'stage2_ops.h3_ltx_adapter')
    base, compat = helpers.public_refiner_modules(paths['ltx_root'])
    cache_api = runtime(paths['sana_root'], 'prompt_cache')
    cache = cache_api.load_cache(paths['prompt_cache'], prompt=cache_api.FIXED_PROMPT, torch_module=torch)
    session = SimpleNamespace(torch=torch, dist=dist, paths=paths, cpu_offload=False,
                              compile_enabled=False, payload=payload, prompt_cache=cache)
    Parent = official._refiner_class(base, compat, session)
    class Refiner(Parent):
        def __init__(self):
            # Only residency and parallel setup differ from the upstream constructor.
            self.rank, self.world_size, self.device = engine.rank, engine.world_size, engine.device
            self.dtype, self.compile_enabled = torch.bfloat16, False
            self.attention_backend = self.sol_backend = 'dense'
            self.stage2_audio_enabled, self.retain_allocator_cache = True, True
            self.stage2_sigmas = helpers.SIGMAS
            self.sigmas = torch.tensor(helpers.SIGMAS, device=self.device, dtype=torch.float32)
            self.h3_upscaler_calls = self.adapter_convert_calls = 0
            self.last_decoder_stream_info = None
            pixel = base.VideoPixelShape(batch=1, frames=121, height=768, width=1344, fps=24)
            self.video_tools = base.VideoLatentTools(base.VideoLatentPatchifier(patch_size=1),
                base.VideoLatentShape.from_pixel_shape(pixel, scale_factors=base.VIDEO_SCALE_FACTORS),
                24., scale_factors=base.VIDEO_SCALE_FACTORS)
            self.audio_tools = compat.AudioLatentTools(compat.AudioPatchifier(patch_size=1),
                compat.AudioLatentShape.from_video_pixel_shape(pixel))
            self.tiling_config = compat.TileSizeConfig(**{
                k:compat.DimensionSizeConfig(**v) for k,v in helpers.TILES.items()})
            self.tiling_config.validate(compat.tiling_scale_factors_for_vae(str(paths['output_video_vae'])), pixel)

        def _checked_attention_stats(self):
            if getattr(self, 'rsi_attention_policy', False):
                return self.sol_attention.stats()
            return base.ResidentRefiner._checked_attention_stats(self)

    model = Refiner()
    instrumentation = getattr(engine, 'instrumentation', None)
    recorder = None if instrumentation is None else instrumentation.stages['ltx']
    phase = (lambda name: nullcontext()) if recorder is None else recorder.phase
    directory = None if evidence_dir is None else Path(evidence_dir)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    print(f'rank={engine.rank} stage=ltx_bridge', flush=True)
    conditioned = None
    with phase('bridge_and_audio_conditioning'):
        if engine.rank == 0:
            model.h3_upscaler, model.h3_spatial_route = helpers.load_upscaler(
                paths['h3_upscaler_source'], paths['h3_upscaler_checkpoint'],
                device=engine.device, torch_module=torch)
            model.h3_ltx_adapter = adapter_api.H3ToLTXAdapter.from_pretrained(
                paths['adapter_dir'], device=engine.device, dtype=model.dtype)
            video = model.video_encode(payload['h3_normalized'].to(engine.device))
            del model.h3_upscaler, model.h3_ltx_adapter
            _release()
            block = compat.AudioConditioner(str(paths['audio_vae']), model.dtype, engine.device,
                registry=base.ModelRegistry(cache_weights=False, cache_models=False),
                alloc_trim_strategy=base.AllocatorTrimStrategy.DEFER)
            model.audio_encoder = block._encoder_builder.build(device=engine.device, dtype=model.dtype).eval().requires_grad_(False)
            audio = model.encode_source_audio(None)
            if directory is not None:
                torch.save(cpu(dict(input=payload, video=video, audio=audio)), directory/'bridge.pt')
            del model.audio_encoder, block
            conditioned = (video, audio)
    with phase('conditioning_transfer'):
        video, audio = _broadcast(conditioned, engine)
    del conditioned
    video_state, audio_state, generator = model.prepare_multimodal_states(video, audio, [], seed)
    del video, audio
    video_context, audio_context = model.encode_prompt_multimodal(cache_api.FIXED_PROMPT)
    if directory is not None and engine.rank == 0:
        torch.save(cpu(dict(video=video_state, audio=audio_state, video_context=video_context,
                            audio_context=audio_context, sigmas=model.sigmas)), directory/'initial.pt')
    _release()
    dist.barrier(group=engine.cpu_group)
    print(f'rank={engine.rank} stage=load_ltx_fsdp', flush=True)
    policy = None
    if recorder is not None:
        from .ltx_attention_policy import build_video_attention_policy
        # Choose the editable local video policy before model construction.
        # dense_ops remains trusted and fixes every non-video attention path.
        policy = build_video_attention_policy(sdpa_backend=attention_backend)
    with phase('load_transformer'):
        model.transformer, model.attention_manager = load_transformer(paths, base, engine, evidence_dir=directory,
            attention_backend=attention_backend, linear_mode=linear_mode)
    if policy is None:
        model.sol_attention = base.Stage2DenseAttention(model.transformer)
    else:
        from .ltx_policy_guard import CheckedVideoAttentionPolicy
        # Fixed T2V refinement passes [] image conditions above, hence no added
        # prefix tokens. Any future nonzero prefix must use the guard's full
        # dense path; audio/text/cross and masked module callables stay intact.
        model.sol_attention = CheckedVideoAttentionPolicy(model.transformer, policy,
            full_attention_prefix_tokens=0)
        model.rsi_attention_policy = True
        recorder.attention['selected_policy_backend'] = model.sol_attention.backend
    model.wrapped_transformer = base.BatchSplitAdapter(model.transformer, max_batch_size=1)
    captures = []
    if directory is not None:
        # Capture at the global SP boundary, before X0 conversion, plus the X0 output.
        captures = [PredictionCapture(model.transformer.velocity_model, directory, label='raw', trace=True),
                    PredictionCapture(model.transformer, directory, label='x0')]
    print(f'rank={engine.rank} stage=ltx_three_updates', flush=True)
    stepper = base.EulerDiffusionStep()
    if directory is not None:
        stepper = ObservedEuler(stepper, directory)
    model.sol_attention.begin_denoise()
    forward_capture = None
    try:
        if recorder is not None:
            if len(model.sigmas) != 4:
                raise RuntimeError('seven_step/ltx requires exactly four scheduler points')
            # X0Model encloses the SP wrapper and its FSDP target: capture begins
            # before either wrapper invokes pre-forward unsharding collectives.
            forward_capture = recorder.attach(model.transformer)
            recorder.fingerprint('initial_video_latents', video_state.latent)
            recorder.fingerprint('initial_audio_latents', audio_state.latent)
        with phase('denoise'):
            video_state, audio_state = compat.euler_denoising_loop(sigmas=model.sigmas,
                video_state=video_state, audio_state=audio_state, stepper=stepper,
                transformer=model.wrapped_transformer,
                denoiser=compat.SimpleDenoiser(v_context=video_context, a_context=audio_context))
        stats = model._checked_attention_stats()
        if forward_capture is not None:
            forward_capture.close(expected=3)
            recorder.attention.update(observed=stats)
            recorder.fingerprint('final_video_latents', video_state.latent)
            recorder.fingerprint('final_audio_latents', audio_state.latent)
        for capture in captures:
            capture.close(3)
    finally:
        if forward_capture is not None:
            forward_capture.close()
        for capture in captures:
            capture.close()
        captures.clear()
    if directory is not None:
        (directory/f'attention-rank{engine.rank}.json').write_text(json.dumps(stats, indent=2))
    video_latent = model.video_tools.unpatchify(model.video_tools.clear_conditioning(video_state)).latent
    if directory is not None and engine.rank == 0:
        torch.save(cpu(dict(video=video_state, audio=audio_state, video_vae_input=video_latent)), directory/'final.pt')
    weak = weakref.ref(model.transformer)
    del model.wrapped_transformer, model.sol_attention, model.transformer, model.attention_manager
    del video_state, audio_state, video_context, audio_context
    _release()
    if weak() is not None:
        raise RuntimeError('LTX DiT survived into decoding')
    dist.barrier(group=engine.cpu_group)
    print(f'rank={engine.rank} stage=ltx_dit_released', flush=True)
    result = None
    if engine.rank == 0:
        with phase('load_video_vae'):
            decoder = compat.VideoDecoder(str(paths['output_video_vae']), model.dtype, engine.device,
                registry=base.ModelRegistry(cache_weights=False, cache_models=False),
                alloc_trim_strategy=base.AllocatorTrimStrategy.DEFER)
            model.video_decoder = decoder._decoder_builder.build(device=engine.device, dtype=model.dtype).eval().requires_grad_(False)
        if type(model.video_decoder).__name__ != 'ConvVideoDecoder':
            raise RuntimeError('Expected the released ConvVideoDecoder')
        phases = {}
        output = Path(output)
        if collect_raw:
            from .seven_step_output import collect_output
            with phase('video_decode_and_raw_packaging'):
                result = collect_output(model.stream_full_vae(video_latent, generator, phases),
                    payload['audio'], model.write_full_vae_video, torch_module=torch)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            model.write_full_vae_video(model.stream_full_vae(video_latent, generator, phases), 8, None, output)
        if directory is not None:
            (directory/'decode.json').write_text(json.dumps(dict(phases=phases,
                writer=session.writer_chunk_receipt, chunks=model.last_decoder_stream_info), indent=2))
        del model.video_decoder, decoder
    del model, video_latent
    _release()
    dist.barrier(group=engine.cpu_group)
    return result
