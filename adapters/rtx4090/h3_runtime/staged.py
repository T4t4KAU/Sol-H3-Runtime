"""Optional component residency stages around the official modular pipeline."""

from __future__ import annotations

import dataclasses
from contextlib import nullcontext
from datetime import timedelta
from functools import wraps
import gc
from pathlib import Path
import weakref

import torch
import torch.distributed as dist
from diffusers import ModularPipeline
from diffusers.hooks import apply_group_offloading

from .fsdp import load_transformer


def _wrap(block, handler):
    original = type(block).__call__

    def call(self, components, state, *args, **kwargs):
        return handler(original, self, components, state, *args, **kwargs)

    block.__class__ = type("Staged" + type(block).__name__, (type(block),),
                           {"__call__": call, "__module__": __name__})


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value, **{f.name: _cpu_copy(getattr(value, f.name))
                                            for f in dataclasses.fields(value)})
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(v) for v in value)
    return value


def _collect():
    gc.collect()
    torch.cuda.empty_cache()


def _timed_stage(name):
    def decorate(function):
        @wraps(function)
        def call(self, *args, **kwargs):
            with self.phase(name):
                return function(self, *args, **kwargs)
        return call
    return decorate


def initialize(engine, model_path, adapter_path):
    model_path = Path(model_path).expanduser().resolve()
    adapter_path = None if adapter_path is None else Path(adapter_path).expanduser().resolve()
    engine.transformer_component = "transformer_ref" if engine.task == "ref2va" else "transformer"
    if (not (model_path / engine.transformer_component / "config.json").is_file()
            or (adapter_path is not None and not adapter_path.is_file())):
        raise FileNotFoundError(f"Pass a local MiniMax-H3 {engine.transformer_component} directory and adapter file")
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=engine.device, timeout=timedelta(minutes=30))
        engine._owns_process_group = True
    engine.rank, engine.world_size = dist.get_rank(), dist.get_world_size()
    engine.cpu_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
    engine.transformer = None
    engine._adaln_checked = False
    # ComponentsManager retains component references; this instance owns all lifetimes.
    if engine.task == "ref2va":
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Ref2VABlocks

        engine.pipe = MiniMaxH3Ref2VABlocks().init_pipeline(str(model_path))
    else:
        engine.pipe = ModularPipeline.from_pretrained(str(model_path))
    if engine.pipe._components_manager is not None:
        raise RuntimeError("Staged execution requires an unshared pipeline")
    cls = type(engine.pipe)
    device = engine.device
    engine.pipe.__class__ = type("Staged" + cls.__name__, (cls,),
                                {"_execution_device": property(lambda self: device), "__module__": __name__})
    engine.pipe.load_components(names=["scheduler", "audio_scheduler"],
                                pretrained_model_name_or_path=str(model_path))
    if engine.pipe.scheduler is None or engine.pipe.audio_scheduler is None:
        raise RuntimeError("Local schedulers could not be loaded")
    engine.pipe.scheduler.set_shift(12.0)
    engine.pipe.audio_scheduler.set_shift(3.0)
    engine._stages = Stages(engine, model_path, adapter_path)


class Stages:
    def __init__(self, engine, model_path, adapter_path):
        self.engine, self.model_path, self.adapter_path = engine, model_path, adapter_path
        self.rank, self.group = engine.rank, engine.cpu_group
        self.closed = False
        blocks = engine.pipe._blocks.sub_blocks
        _wrap(blocks["text_encoder"], self.text)
        if engine.task == "ref2va":
            _wrap(blocks["reference_encoder"], self.condition)
        else:
            _wrap(blocks["vae_encoder"].sub_blocks["keyframes"], self.condition)
        _wrap(blocks["prepare_layout"], self.rank_zero)
        _wrap(blocks["prepare_latents"], self.rank_zero)
        _wrap(blocks["denoise"], self.denoise)
        _wrap(blocks["decode"].sub_blocks["video"], self.video)
        _wrap(blocks["decode"].sub_blocks["audio"], self.audio)

    def mark(self, stage):
        print(f"rank={self.rank} stage={stage}", flush=True)

    def phase(self, name):
        recorder = self.engine.instrumentation
        return recorder.phase(name) if recorder is not None else nullcontext()

    def sync_state(self, state):
        payload = [None]
        if self.rank == 0:
            values = {k: v for k, v in state.values.items() if k != "generator"}
            cuda_keys = [k for k, v in values.items() if isinstance(v, torch.Tensor) and v.is_cuda]
            payload[0] = (_cpu_copy(values), cuda_keys)
        dist.broadcast_object_list(payload, 0, group=self.group)
        values, cuda_keys = payload[0]
        for key in cuda_keys:
            values[key] = values[key].to(self.engine.device)
        # Layout dataclasses and tags keep their original CPU placement.
        state.values.update(values)

    def rank_zero(self, original, block, components, state):
        if self.rank == 0:
            components, state = original(block, components, state)
        self.sync_state(state)
        return components, state

    @_timed_stage('text_stage')
    def text(self, original, block, components, state):
        self.mark("text_encode")
        if self.engine.runtime.text_fsdp:
            from .text_fsdp import load_text_encoder

            with self.phase('load_text_encoder'):
                components.load_components(names=["tokenizer", "processor"],
                                           pretrained_model_name_or_path=str(self.model_path))
                model = load_text_encoder(self.engine, self.model_path / "text_encoder")
            components.register_components(text_encoder=model)
            # Every rank calls the official encoder, which enters text_encoder.model.
            def check_inputs(_model, args, kwargs):
                from .distributed_loading import tensor_digest
                from .evidence import cpu
                digest = {k:tensor_digest(v) for k,v in kwargs.items() if isinstance(v,torch.Tensor)}
                received = [None]*self.engine.world_size
                dist.all_gather_object(received, digest, group=self.group)
                if any(item != digest for item in received):
                    raise RuntimeError('Qwen ranks received different input tokens')
                if self.rank == 0 and self.engine.runtime.evidence_dir:
                    path = Path(self.engine.runtime.evidence_dir)
                    path.mkdir(parents=True, exist_ok=True)
                    torch.save(cpu(dict(args=args, inputs=kwargs)), path/'qwen-inputs.pt')
            handle = model.model.register_forward_pre_hook(check_inputs, with_kwargs=True)
            try:
                with self.phase('text_encode'):
                    components, state = original(block, components, state)
            finally:
                handle.remove()
            expected = state.prompt_embeds.clone()
            dist.broadcast(expected, 0)
            if not torch.equal(expected, state.prompt_embeds):
                raise RuntimeError("Qwen embeddings differ across ranks")
            if self.rank == 0 and self.engine.runtime.evidence_dir:
                torch.save(state.prompt_embeds.detach().cpu(),
                           Path(self.engine.runtime.evidence_dir)/'qwen-prompt-embeds.pt')
            components.register_components(text_encoder=None, tokenizer=None, processor=None)
            del model, expected
            _collect()
            self.sync_state(state)
            return components, state
        if self.rank == 0:
            with self.phase('load_text_encoder'):
                components.load_components(names=["text_encoder", "tokenizer", "processor"],
                                            dtype=torch.bfloat16, pretrained_model_name_or_path=str(self.model_path))
            if any(getattr(components, n) is None for n in ("text_encoder", "tokenizer", "processor")):
                raise RuntimeError("Local text components could not be loaded")
            components.text_encoder.eval().requires_grad_(False)
            apply_group_offloading(components.text_encoder, onload_device=self.engine.device,
                                   offload_type="leaf_level", use_stream=False)
            with self.phase('text_encode'):
                components, state = original(block, components, state)
            components.register_components(text_encoder=None, tokenizer=None, processor=None)
            _collect()
        self.sync_state(state)
        return components, state

    @_timed_stage('denoise_stage')
    def denoise(self, original, block, components, state):
        if not len(state.timesteps) or len(state.timesteps) != len(state.audio_timesteps):
            raise RuntimeError("Video and audio denoising schedules must have equal nonzero lengths")
        self.mark("load_transformer")
        with self.phase('load_transformer'):
            model = load_transformer(self.model_path, self.adapter_path, self.group,
                                     state.timesteps, state.audio_timesteps, engine=self.engine)
        component = self.engine.transformer_component
        components.register_components(**{component: model})
        self.engine.transformer = model
        self.engine._adaln_checked = False
        evidence = None
        capture = None
        recorder = self.engine.instrumentation
        self.mark("denoise")
        try:
            if recorder is not None:
                if len(state.timesteps) != recorder.expected_forwards:
                    raise RuntimeError(f'{recorder.label} instrumentation requires exactly '
                                       f'{recorder.expected_forwards} H3 denoising timesteps')
                recorder.fingerprint('initial_video_latents', state.latents)
                recorder.fingerprint('initial_audio_latents', state.audio_latents)
                capture = recorder.attach(model)
            if self.engine.runtime.evidence_dir:
                from .evidence import H3Evidence
                evidence = H3Evidence(model, block, components, state,
                                      self.engine.runtime.evidence_dir, self.group)
            with self.phase('denoise'):
                components, state = original(block, components, state)
            if capture is not None:
                capture.close(expected=recorder.expected_forwards)
                recorder.audit_attention(getattr(self.engine, 'sparse_attention', None))
                recorder.fingerprint('final_video_latents', state.latents)
                recorder.fingerprint('final_audio_latents', state.audio_latents)
            if evidence is not None:
                evidence.close(state)
        finally:
            try:
                if capture is not None:
                    capture.close()
            finally:
                if evidence is not None:
                    evidence.close()
            del evidence, capture
        self.engine._check_adaln()
        self.mark("release_transformer")
        weak = weakref.ref(model)
        self.engine.transformer = None
        # The official LoRA fusion state owns a list of projection modules.
        self.engine.lora_fusion = None
        self.engine.sparse_attention = None
        components.register_components(**{component: None})
        del model
        _collect()
        if weak() is not None:
            raise RuntimeError("DiT reference survived into the decoding phase")
        dist.barrier(group=self.group)
        self.mark("transformer_released")
        return components, state

    @_timed_stage('condition_stage')
    def condition(self, original, block, components, state):
        """Keep the official I2V/Ref2VA encoders, with VAEs resident only for conditioning."""
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

        self.mark("condition_encode")
        vae = AutoencoderKLMiniMaxH3.from_pretrained(str(self.model_path / "vae"),
                torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False).to(self.engine.device)
        vae.enable_tiling()
        components.register_components(vae=vae)
        reference = self.engine.task == "ref2va"
        undo = self.engine._install_vae(vae, encode_parallel=reference)
        audio_vae = None
        if reference:
            audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(self.model_path / "audio_vae"),
                    torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False).to(self.engine.device)
            components.register_components(audio_vae=audio_vae)
        try:
            components, state = original(block, components, state)
        finally:
            undo()
            components.register_components(vae=None)
            if reference:
                components.register_components(audio_vae=None)
            del vae, audio_vae, undo
            _collect()
        self.sync_state(state)
        return components, state

    @_timed_stage('video_stage')
    def video(self, original, block, components, state):
        if self.engine._capture_latents:
            from diffusers.modular_pipelines.minimax_h3.packing import unpatchify_video_tokens

            if self.rank == 0:
                value = block.get_block_state(state)
                native = unpatchify_video_tokens(
                    value.latents[value.num_condition_video_rows:], value.num_latent_frames,
                    value.latent_height, value.latent_width, components.vae_latent_channels,
                    components.patch_size,
                ).detach().cpu().clone()
                if native.dtype != torch.float32 or not torch.isfinite(native).all():
                    raise RuntimeError("Expected finite FP32 normalized H3 latent before VAE denormalization")
                state.values["h3_normalized_native"] = native
            self.mark("video_latent_capture")
            return components, state
        from diffusers import AutoencoderKLMiniMaxH3

        self.mark("video_decode_load")
        if not self.engine.runtime.parallel_vae and self.rank != 0:
            dist.barrier(group=self.group)
            return components, state
        with self.phase('load_video_vae'):
            vae = AutoencoderKLMiniMaxH3.from_pretrained(str(self.model_path / "vae"),
                    torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False).to(self.engine.device)
        vae.enable_tiling()
        components.register_components(vae=vae)
        undo = self.engine._install_vae(vae)
        self.mark("video_decode")
        from .evidence import capture_decode
        with self.phase('video_decode'):
            with capture_decode(vae, self.engine.runtime.evidence_dir if self.rank == 0 else None, 'video'):
                components, state = original(block, components, state)
        if self.rank == 0:
            videos = state.videos
            tensors = [videos] if isinstance(videos, torch.Tensor) else videos
            if not tensors or not all(isinstance(v, torch.Tensor) and v.is_cuda for v in tensors):
                raise RuntimeError("The official video encoder requires CUDA video output")
        else:
            state.values["videos"] = None
        undo()
        components.register_components(vae=None)
        del vae, undo
        _collect()
        dist.barrier(group=self.group)
        return components, state

    @_timed_stage('audio_stage')
    def audio(self, original, block, components, state):
        from diffusers import AutoencoderKLMiniMaxH3Audio

        self.mark("audio_decode")
        if self.rank == 0:
            with self.phase('load_audio_vae'):
                vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(self.model_path / "audio_vae"),
                        torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False).to(self.engine.device)
            components.register_components(audio_vae=vae)
            from .evidence import capture_decode
            with self.phase('audio_decode'):
                with capture_decode(vae, self.engine.runtime.evidence_dir, 'audio'):
                    components, state = original(block, components, state)
            state.values["audio"] = _cpu_copy(state.audio)
            components.register_components(audio_vae=None)
            del vae
            _collect()
        dist.barrier(group=self.group)
        return components, state

    def close(self):
        if self.closed:
            return
        self.engine.transformer = None
        self.engine.lora_fusion = None
        self.engine.sparse_attention = None
        self.engine.pipe.register_components(**{self.engine.transformer_component: None},
                                              text_encoder=None, tokenizer=None, processor=None,
                                              vae=None, audio_vae=None)
        _collect()
        if dist.is_initialized():
            dist.destroy_process_group(self.group)
        self.closed = True
