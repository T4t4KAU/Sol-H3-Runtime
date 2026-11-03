"""Resident H3(4) -> latent bridge -> LTX(3) service on one 80-GiB H800."""

from __future__ import annotations

from contextlib import contextmanager
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

from resident_h800 import H800ResidentBackend


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Expected seven-step configuration schema_version=1")
    assets = Path(config["assets_root"]).resolve()
    ltx = assets / "LTX-2.5"
    paths = dict(
        sana_root=config["sana_root"],
        ltx_root=config["ltx_root"],
        transformer=str(
            ltx / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
        ),
        refiner_lora=str(ltx / "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"),
        audio_vae=str(ltx / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
        output_video_vae=str(ltx / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
        adapter_dir=str(assets / "H3-to-LTX-Latent-Adapter"),
        h3_upscaler_source=config["h3_upscaler_source"],
        h3_upscaler_checkpoint=str(
            assets / "H3-upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors"
        ),
        prompt_cache=str(assets / "ltx-fixed.pt"),
    )
    return config, paths


def dependencies(config, paths):
    extra = config.get("extra_python_path")
    if extra:
        sys.path.insert(0, str(Path(extra).resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters/rtx4090"))
    from h3_runtime.upstream import ltx_paths, runtime

    ltx_paths(paths["ltx_root"])
    # The upstream Python module imports the optional multi-GPU extension eagerly.
    # This backend uses NCCL's singleton transport instead, never the C++ kernel.
    if (
        "all2all_cpp" not in sys.modules
        and importlib.util.find_spec("all2all_cpp") is None
    ):
        module = types.ModuleType("all2all_cpp")

        class UnavailableMultiGPUTransport:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "Compiled LTX IPC transport is unavailable; use the singleton NCCL manager"
                )

        module.All2All = UnavailableMultiGPUTransport
        sys.modules["all2all_cpp"] = module
    helpers = runtime(paths["sana_root"], "stage2_ops.models")
    base, compat = helpers.public_refiner_modules(paths["ltx_root"])
    return runtime, helpers, base, compat


class H800SevenStepBackend(H800ResidentBackend):
    def __init__(self, model, adapter, config_path):
        started = time.monotonic()
        config, self.paths = read_config(config_path)
        # Import dependencies and validate all path identities before allocating models.
        runtime, helpers, base, compat = dependencies(config, self.paths)
        self.refiner = None
        self.stage2_objects = {}
        self.forward_handles = []
        self.forward_counts = {"h3": 0, "ltx": 0}
        self.base, self.compat = base, compat
        self.cache_api = runtime(self.paths["sana_root"], "prompt_cache")
        import torch

        self.prompt_cache = self.cache_api.load_cache(
            self.paths["prompt_cache"],
            prompt=self.cache_api.FIXED_PROMPT,
            torch_module=torch,
        )
        for name in [
            "transformer",
            "refiner_lora",
            "audio_vae",
            "output_video_vae",
            "h3_upscaler_source",
            "h3_upscaler_checkpoint",
        ]:
            if not Path(self.paths[name]).is_file():
                raise FileNotFoundError(self.paths[name])
        super().__init__(model, adapter)
        try:
            self._place_h3("cpu")
            official = runtime(self.paths["sana_root"], "stage2")
            session = types.SimpleNamespace(
                torch=torch,
                dist=__import__("torch.distributed", fromlist=[""]),
                paths=self.paths,
                cpu_offload=False,
                compile_enabled=False,
                payload=None,
                prompt_cache=self.prompt_cache,
            )
            self.session = session
            Parent = official._refiner_class(base, compat, session)

            class Refiner(Parent):
                def __init__(inner):
                    inner.rank, inner.world_size, inner.device = (
                        0,
                        1,
                        torch.device("cuda", 0),
                    )
                    inner.dtype, inner.compile_enabled = torch.bfloat16, False
                    inner.attention_backend = inner.sol_backend = "dense"
                    inner.stage2_audio_enabled, inner.retain_allocator_cache = (
                        True,
                        True,
                    )
                    inner.stage2_sigmas = helpers.SIGMAS
                    inner.sigmas = torch.tensor(
                        helpers.SIGMAS, device=inner.device, dtype=torch.float32
                    )
                    inner.h3_upscaler_calls = inner.adapter_convert_calls = 0
                    inner.last_decoder_stream_info = None
                    pixel = base.VideoPixelShape(
                        batch=1, frames=121, height=768, width=1344, fps=24
                    )
                    inner.video_tools = base.VideoLatentTools(
                        base.VideoLatentPatchifier(patch_size=1),
                        base.VideoLatentShape.from_pixel_shape(
                            pixel, scale_factors=base.VIDEO_SCALE_FACTORS
                        ),
                        24.0,
                        scale_factors=base.VIDEO_SCALE_FACTORS,
                    )
                    inner.audio_tools = compat.AudioLatentTools(
                        compat.AudioPatchifier(patch_size=1),
                        compat.AudioLatentShape.from_video_pixel_shape(pixel),
                    )
                    inner.tiling_config = compat.TileSizeConfig(
                        **{
                            k: compat.DimensionSizeConfig(**v)
                            for k, v in helpers.TILES.items()
                        }
                    )
                    inner.tiling_config.validate(
                        compat.tiling_scale_factors_for_vae(
                            self.paths["output_video_vae"]
                        ),
                        pixel,
                    )

                def _checked_attention_stats(inner):
                    return base.ResidentRefiner._checked_attention_stats(inner)

            ref = self.refiner = Refiner()
            from h3_runtime.two_stage import transformer_builder
            from h3_runtime.ltx_sp import NCCLAttentionManager, require_divisible_video
            from ltx_core.loader.fuse_loras import FuseRule
            from ltx_core.model.transformer.model import X0Model
            from ltx_core.multigpu.transformer.sequence_parallel import (
                SequenceParallelModelWrapper,
                create_video_self_attention_module_ops,
            )

            self.stages.mark("preload_ltx_transformer")
            manager = NCCLAttentionManager(session.dist.group.WORLD)
            builder = transformer_builder(self.paths, base, self.engine.device)
            rule = builder.fuse_rule

            def retain_cpu(key, weight, deltas, state):
                return {k: v.cpu() for k, v in rule(key, weight, deltas, state).items()}

            builder = builder.with_fuse_rule(
                FuseRule(rule.aggregation_dtype, retain_cpu)
            )
            builder = builder.with_module_ops(
                (*builder.module_ops, create_video_self_attention_module_ops(manager))
            )
            core = (
                builder.build(device=torch.device("cpu"), dtype=torch.bfloat16)
                .eval()
                .requires_grad_(False)
            )
            if len(core.transformer_blocks) != 48:
                raise RuntimeError("Expected 48 LTX blocks")
            wrapped = SequenceParallelModelWrapper(core, manager)
            wrapped.register_forward_pre_hook(require_divisible_video, with_kwargs=True)
            ref.transformer = X0Model(wrapped).eval().requires_grad_(False)
            ref.attention_manager = manager
            ref.sol_attention = base.Stage2DenseAttention(ref.transformer)
            ref.wrapped_transformer = base.BatchSplitAdapter(
                ref.transformer, max_batch_size=1
            )
            self.stages.mark("preload_ltx_bridge_and_vaes")
            ref.h3_upscaler, ref.h3_spatial_route = helpers.load_upscaler(
                self.paths["h3_upscaler_source"],
                self.paths["h3_upscaler_checkpoint"],
                device=self.engine.device,
                torch_module=torch,
            )
            ref.h3_upscaler.cpu()
            adapter_api = runtime(self.paths["sana_root"], "stage2_ops.h3_ltx_adapter")
            ref.h3_ltx_adapter = adapter_api.H3ToLTXAdapter.from_pretrained(
                self.paths["adapter_dir"], device=torch.device("cpu"), dtype=ref.dtype
            )
            registry = base.ModelRegistry(cache_weights=False, cache_models=False)
            audio = compat.AudioConditioner(
                self.paths["audio_vae"],
                ref.dtype,
                self.engine.device,
                registry=registry,
                alloc_trim_strategy=base.AllocatorTrimStrategy.DEFER,
            )
            ref.audio_encoder = (
                audio._encoder_builder.build(
                    device=torch.device("cpu"), dtype=ref.dtype
                )
                .eval()
                .requires_grad_(False)
            )
            decoder = compat.VideoDecoder(
                self.paths["output_video_vae"],
                ref.dtype,
                self.engine.device,
                registry=registry,
                alloc_trim_strategy=base.AllocatorTrimStrategy.DEFER,
            )
            ref.video_decoder = (
                decoder._decoder_builder.build(
                    device=torch.device("cpu"), dtype=ref.dtype
                )
                .eval()
                .requires_grad_(False)
            )
            if type(ref.video_decoder).__name__ != "ConvVideoDecoder":
                raise RuntimeError("Expected official ConvVideoDecoder")
            self.stage2_objects = dict(
                ltx_transformer=ref.transformer,
                ltx_audio_encoder=ref.audio_encoder,
                ltx_video_decoder=ref.video_decoder,
                h3_upscaler=ref.h3_upscaler,
                h3_ltx_adapter=ref.h3_ltx_adapter.model,
            )
            self.stage2_ids = {n: id(m) for n, m in self.stage2_objects.items()}

            def count(stage):
                def hook(*_args):
                    self.forward_counts[stage] += 1

                return hook

            self.forward_handles = [
                self.engine.transformer.register_forward_pre_hook(count("h3")),
                core.register_forward_pre_hook(count("ltx")),
            ]
            self._place_h3(self.engine.device)
            torch.cuda.synchronize()
            self.stats.update(
                backend="h800-seven-step",
                generation_mode="H3(4)+LTX(3)",
                placement="Models retained in memory; H3 and LTX stages swap between CPU and GPU",
                startup_s=time.monotonic() - started,
                component_ids={**self.component_ids, **self.stage2_ids},
                component_load_counts={
                    n: 1 for n in [*self.component_ids, *self.stage2_ids]
                },
                output_profile=dict(width=1344, height=768, frames=121, fps=24),
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
            )
            self.stages.mark("seven_step_resident_ready")
        except BaseException:
            self.close()
            raise

    def _place_h3(self, device):
        import torch

        for name in ("transformer", "vae", "audio_vae"):
            getattr(self.engine.pipe, name).to(device)
        gc.collect()
        torch.cuda.empty_cache()

    def _check_identity(self):
        for name, identity in self.component_ids.items():
            if id(getattr(self.engine.pipe, name)) != identity:
                raise RuntimeError("Replaced H3 component: " + name)
        for name, identity in self.stage2_ids.items():
            if id(self.stage2_objects[name]) != identity:
                raise RuntimeError("Replaced LTX component: " + name)

    def generate(self, parameters, output, log):
        import torch

        with torch.no_grad():
            return self._generate(parameters, output, log)

    def _generate(self, parameters, output, log):
        import torch

        started = time.monotonic()
        self.request_log = log
        intervals = []
        self.forward_counts = {"h3": 0, "ltx": 0}

        @contextmanager
        def phase(name):
            begin = time.monotonic() - started
            self.stages.mark(name)
            torch.cuda.synchronize()
            try:
                yield
            finally:
                torch.cuda.synchronize()
                intervals.append(
                    dict(name=name, start_s=begin, end_s=time.monotonic() - started)
                )

        try:
            torch.cuda.reset_peak_memory_stats()
            self._check_identity()
            with phase("h3_generation"):
                media = self.engine.generate(
                    parameters["prompt"],
                    duration=5,
                    seed=parameters["seed"],
                    width=672,
                    height=384,
                    capture_latents=True,
                    num_inference_steps=5,
                )
                if (
                    media.audio_sample_rate != 32000
                    or media.audio.dtype != torch.float32
                    or media.audio.ndim != 3
                    or tuple(media.audio.shape[:2]) != (1, 2)
                    or media.audio.shape[2] < 161333
                    or not bool(torch.isfinite(media.audio).all())
                ):
                    raise RuntimeError("Expected H3 stereo PCM at 32 kHz")
                self.session.payload = dict(
                    h3_normalized=media.video.to(torch.bfloat16).contiguous(),
                    audio=media.audio[:, :, :161333].clamp(-1, 1).contiguous(),
                )
                del media
            with phase("park_h3"):
                self._place_h3("cpu")
            ref = self.refiner
            with phase("bridge_and_audio_conditioning"):
                ref.h3_upscaler.to(self.engine.device)
                ref.h3_ltx_adapter.model.to(self.engine.device)
                ref.h3_ltx_adapter.device = self.engine.device
                video = ref.video_encode(
                    self.session.payload["h3_normalized"].to(self.engine.device)
                )
                ref.h3_upscaler.cpu()
                ref.h3_ltx_adapter.model.cpu()
                ref.h3_ltx_adapter.device = torch.device("cpu")
                ref.audio_encoder.to(self.engine.device)
                audio = ref.encode_source_audio(None)
                ref.audio_encoder.cpu()
                video_state, audio_state, generator = ref.prepare_multimodal_states(
                    video, audio, [], parameters["seed"]
                )
                del video, audio
                video_context, audio_context = ref.encode_prompt_multimodal(
                    self.cache_api.FIXED_PROMPT
                )
                gc.collect()
                torch.cuda.empty_cache()
            with phase("load_ltx_to_gpu"):
                ref.transformer.to(self.engine.device)
            ref.sol_attention.begin_denoise()
            with phase("ltx_three_updates"):
                video_state, audio_state = self.compat.euler_denoising_loop(
                    sigmas=ref.sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=self.base.EulerDiffusionStep(),
                    transformer=ref.wrapped_transformer,
                    denoiser=self.compat.SimpleDenoiser(
                        v_context=video_context, a_context=audio_context
                    ),
                )
                attention = ref._checked_attention_stats()
                if self.forward_counts != {"h3": 4, "ltx": 3}:
                    raise RuntimeError(
                        "Wrong seven-step counts: " + str(self.forward_counts)
                    )
                latent = ref.video_tools.unpatchify(
                    ref.video_tools.clear_conditioning(video_state)
                ).latent
                del video_state, audio_state, video_context, audio_context
            with phase("park_ltx"):
                ref.transformer.cpu()
                gc.collect()
                torch.cuda.empty_cache()
            with phase("video_decode_and_encoding"):
                ref.video_decoder.to(self.engine.device)
                ref.write_full_vae_video(
                    ref.stream_full_vae(latent, generator, {}), 8, None, Path(output)
                )
                ref.video_decoder.cpu()
                del latent, generator
                self.session.payload = None
                gc.collect()
                torch.cuda.empty_cache()
            with phase("restore_h3_to_gpu"):
                self._place_h3(self.engine.device)
            self._check_identity()
            self.stats["requests_completed"] += 1
            self.stats.update(
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
            )
            result = dict(
                backend="h800-seven-step",
                worker_pid=os.getpid(),
                model_instance_id=self.stats["model_instance_id"],
                model_load_count=1,
                request_index=self.stats["requests_completed"],
                component_ids=dict(self.stats["component_ids"]),
                forward_counts=dict(self.forward_counts),
                request_wall_s=time.monotonic() - started,
                phases=intervals,
                attention=attention,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                output_profile=dict(width=1344, height=768, frames=121, fps=24),
            )
            (Path(output).parent / "seven_step_receipt.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            log.write(
                (
                    json.dumps(dict(event="seven_step_complete", **result)) + "\n"
                ).encode()
            )
            log.flush()
            return result
        except BaseException as exc:
            self.fatal_error = f"Seven-step request failed: {type(exc).__name__}: {exc}"
            failure = dict(
                status="failed",
                error=self.fatal_error,
                traceback=traceback.format_exc(),
                phases=intervals,
                forward_counts=self.forward_counts,
                request_wall_s=time.monotonic() - started,
            )
            (Path(output).parent / "seven_step_failure.json").write_text(
                json.dumps(failure, indent=2) + "\n"
            )
            log.write((failure["traceback"] + "\n").encode())
            log.flush()
            raise
        finally:
            self.request_log = None

    def close(self):
        for handle in getattr(self, "forward_handles", []):
            handle.remove()
        self.forward_handles = []
        self.stage2_objects = {}
        self.refiner = None
        if hasattr(self, "engine"):
            super().close()
