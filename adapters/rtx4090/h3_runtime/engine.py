"""Resident 768p MiniMax-H3 T2V/I2V/Ref2VA inference engine."""

from __future__ import annotations

import math
import gc
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel
from PIL import Image, ImageOps

from .compute_quant import COMPUTE_QUANT_MODES


WIDTH = 1344
HEIGHT = 768
FPS = 24
INFERENCE_STEPS = 5  # Five scheduler points execute four DiT forwards.
DURATION_FRAMES = {5: 124, 10: 243, 15: 362}
ATTENTION_BACKENDS = {"dense", "sol", "sol_bsa"}
TASKS = {"t2v", "i2v", "ref2va"}
REFERENCE_IMAGE_RESIZE_MODES = {"match", "diffusers"}


def _resolve_reference_image_size(
    width: int,
    height: int,
    *,
    mode: str,
) -> tuple[int, int]:
    """Resolve Ref2VA image geometry for the selected deployment profile."""
    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {width}x{height}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")

    if mode == "match":
        # Match the fastest validated Ref2VA profile: never upscale a source,
        # and cap larger inputs to the generated 768p canvas area.
        scale = min(1.0, math.sqrt((WIDTH * HEIGHT) / (width * height)))
    elif mode == "diffusers":
        # Official Diffusers parity mode. This intentionally upscales every
        # reference image to a 2048-pixel short edge.
        scale = 2048 / min(width, height)
    else:
        raise ValueError(
            f"reference_image_resize_mode must be one of {sorted(REFERENCE_IMAGE_RESIZE_MODES)}"
        )

    multiple = 32
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def _state_value(state: Any, name: str) -> Any:
    if hasattr(state, "get"):
        value = state.get(name)
        if value is not None:
            return value
    return getattr(state, name, None)


@dataclass
class GeneratedMedia:
    """Rank-zero output. Other ranks return ``None`` from ``generate``."""

    video: Any
    audio: Any
    audio_sample_rate: int | None
    elapsed_s: float
    duration: int
    seed: int

    def save_async(self, output_path: str | Path):
        """Stage output and encode it in the background."""
        from .encoding import start_fast_video_encode

        if self.video is None:
            raise RuntimeError("this output has already been handed to an encoder")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        job = start_fast_video_encode(
            self.video,
            fps=FPS,
            output_path=str(output_path),
            audio=self.audio,
            audio_sample_rate=self.audio_sample_rate,
            chunk_frames=8,
            fragmented=True,
            encoder_threads=36,
            pyav_zero_copy=True,
            overlap_audio=True,
        )
        job.wait_staged()
        self.video = None
        self.audio = None
        return job

    def save(self, output_path: str | Path) -> Path:
        job = self.save_async(output_path)
        job.wait()
        output_path = Path(output_path)
        return output_path


@dataclass
class GeneratedLatents:
    """Native normalized FP32 video latent and decoded FP32 stereo PCM."""

    video: Any
    audio: Any
    audio_sample_rate: int
    elapsed_s: float
    seed: int


@dataclass(frozen=True)
class RuntimeConfig:
    """Optional execution settings; deployment scripts choose hardware-specific values."""

    fsdp_degree: int | None = None
    ulysses_degree: int | None = None
    staged_loading: bool = False
    flash_sdpa: bool = False
    allow_tf32: bool | None = None
    allow_bf16_reduced_precision_reduction: bool | None = None
    vae_batched: bool | None = None
    vae_global_batch: bool | None = None
    vae_compile_mode: str | None = None  # None inherits H3_VAE_COMPILE; "eager" disables it.
    adaln_precompute: bool = True
    kernel_fusions: bool = True
    packed_sp: bool = True
    text_fsdp: bool = False
    parallel_vae: bool = True
    evidence_dir: str | None = None
    vae_tile_size: int | None = None

    def __post_init__(self):
        for name in ("fsdp_degree", "ulysses_degree"):
            degree = getattr(self, name)
            if degree is not None and (type(degree) is not int or degree < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.vae_compile_mode not in {None, "eager", "default", "max-autotune-no-cudagraphs"}:
            raise ValueError("vae_compile_mode must be eager, default, max-autotune-no-cudagraphs, or None")
        if self.vae_tile_size is not None and (self.vae_tile_size < 128 or self.vae_tile_size % 32):
            raise ValueError('vae_tile_size must be a multiple of 32 and at least 128')


class MiniMaxH3Inference:
    """Load once and generate fixed-profile 768p T2V, I2V, or Ref2VA."""

    def __init__(
        self,
        model_path: str,
        adapter_path: str | Path | None,
        attention_backend: str = "sol_bsa",
        task: str = "t2v",
        reference_image_resize_mode: str = "match",
        compute_quant: str = "none",
        lora_mode: str = "merged",
        *,
        runtime: RuntimeConfig | None = None,
        instrumentation: Any | None = None,
    ) -> None:
        self.runtime = runtime or RuntimeConfig()
        self.instrumentation = instrumentation
        self._stages = None
        self._meshes = {}
        self._mesh_groups = []
        self._closed = False
        if self.runtime.fsdp_degree not in (None, 1) and not self.runtime.staged_loading:
            raise ValueError("FSDP loading requires staged_loading")
        if self.runtime.text_fsdp and not self.runtime.staged_loading:
            raise ValueError("Distributed text loading requires staged_loading")
        if self.runtime.evidence_dir and not self.runtime.staged_loading:
            raise ValueError("Evidence capture requires staged_loading")
        self._capture_latents = False
        lora_mode = lora_mode.lower()
        if lora_mode not in {"merged", "separate", "fused"}:
            raise ValueError("lora_mode must be merged, separate, or fused")
        if lora_mode != "merged" and compute_quant.lower() != "none":
            raise ValueError("Separate/fused LoRA requires BF16 linear compute")
        self.lora_mode = lora_mode
        self.lora_fusion = None
        self._owns_process_group = False
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)

        if dist.is_available() and not dist.is_initialized():
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size > 1:
                # Staged cold loading leaves peers waiting while rank zero encodes text.
                kwargs = {"timeout": timedelta(minutes=30)} if self.runtime.staged_loading else {}
                dist.init_process_group(backend="nccl", device_id=self.device, **kwargs)
                self._owns_process_group = True

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.world_size not in {1, 2, 4, 8}:
            raise ValueError("GPU process count must be 1, 2, 4, or 8")
        self.ulysses_degree = (self.world_size if self.runtime.ulysses_degree is None
                               else self.runtime.ulysses_degree)
        if self.ulysses_degree not in {1, self.world_size}:
            raise ValueError("ulysses_degree must be 1 or the process-group world size")
        if self.runtime.fsdp_degree is not None and self.runtime.fsdp_degree not in {1, self.world_size}:
            raise ValueError("fsdp_degree must be 1 or the process-group world size")
        if self.runtime.allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = self.runtime.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.runtime.allow_tf32
        if self.runtime.allow_bf16_reduced_precision_reduction is not None:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                self.runtime.allow_bf16_reduced_precision_reduction
            )
        task = task.lower()
        if task not in TASKS:
            raise ValueError(f"task must be one of {sorted(TASKS)}")
        attention_backend = attention_backend.lower()
        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"attention_backend must be one of {sorted(ATTENTION_BACKENDS)}"
            )
        if attention_backend != "dense" and self.ulysses_degree == 1:
            raise ValueError("SOL attention requires 2, 4, or 8 GPU processes")
        if task == "ref2va" and attention_backend == "sol":
            raise ValueError("Ref2VA supports dense or sol_bsa attention")
        reference_image_resize_mode = reference_image_resize_mode.lower()
        if reference_image_resize_mode not in REFERENCE_IMAGE_RESIZE_MODES:
            raise ValueError(
                "reference_image_resize_mode must be one of "
                f"{sorted(REFERENCE_IMAGE_RESIZE_MODES)}"
            )
        compute_quant = compute_quant.lower()
        if compute_quant not in COMPUTE_QUANT_MODES:
            raise ValueError(f"compute_quant must be one of {sorted(COMPUTE_QUANT_MODES)}")
        if self.runtime.staged_loading and compute_quant != "none":
            raise ValueError("Staged loading currently requires BF16 compute; MXFP8 uses the resident path")
        if compute_quant == "mxfp8":
            from .compute_quant import validate_platform

            validate_platform()
        self.task = task
        self.attention_backend = attention_backend
        self.reference_image_resize_mode = reference_image_resize_mode
        self.compute_quant = compute_quant

        from diffusers import ComponentsManager, ModularPipeline

        # The released pipeline originally enforced a shorter public limit. The
        # 362-frame shape is a native 17*n+5 H3 sequence and is used directly.
        from diffusers.modular_pipelines.minimax_h3 import before_encoder

        before_encoder.MINIMAX_H3_MAX_DURATION = 364 / FPS
        if task == "ref2va":
            # before_encoder imports the resolver by name, so override that
            # binding without modifying the installed Diffusers package.
            before_encoder.resolve_reference_image_size = lambda width, height: (
                _resolve_reference_image_size(
                    width,
                    height,
                    mode=self.reference_image_resize_mode,
                )
            )

        if self.runtime.staged_loading:
            from .staged import initialize

            try:
                initialize(self, model_path, adapter_path)
            except Exception:
                self.close(synchronize=False)
                raise
            return

        manager = ComponentsManager()
        if task == "ref2va":
            from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Ref2VABlocks

            self.pipe = MiniMaxH3Ref2VABlocks().init_pipeline(
                model_path, components_manager=manager
            )
        else:
            self.pipe = ModularPipeline.from_pretrained(
                model_path, components_manager=manager
            )
        load_kwargs = {"dtype": torch.bfloat16}
        if task == "ref2va":
            # Resolve every Ref2VA component below the selected model root. This
            # also avoids stale absolute paths in locally converted indexes.
            load_kwargs["pretrained_model_name_or_path"] = model_path
        self.pipe.load_components(**load_kwargs)
        self.transformer = (
            self.pipe.transformer_ref if task == "ref2va" else self.pipe.transformer
        )
        if self.transformer is None:
            component = "transformer_ref" if task == "ref2va" else "transformer"
            raise RuntimeError(
                f"failed to load {component} from {model_path!r}; "
                f"make sure {component}/config.json and all checkpoint shards exist"
            )
        self.pipe.to(self.device)
        self.pipe.scheduler.set_shift(12.0)
        self.pipe.audio_scheduler.set_shift(3.0)

        from .lora import fuse_lora, load_lora_branches

        load_adapter = fuse_lora if lora_mode == "merged" else load_lora_branches
        if adapter_path is not None:
            load_adapter(
                self.transformer, adapter_path,
                alpha=8 if task == "ref2va" else 64, scale=1.0,
            )

        if self.ulysses_degree > 1:
            from diffusers.models._modeling_parallel import ContextParallelConfig

            from .cp_plan import MINIMAX_H3_CP_PLAN, assert_no_attention_mask

            assert_no_attention_mask(self.transformer)
            self.transformer.enable_parallelism(
                config=ContextParallelConfig(
                    ulysses_degree=self.ulysses_degree,
                    ring_degree=1,
                    ulysses_anything=True,
                    mesh=self._get_parallel_mesh("cp"),
                ),
                cp_plan=MINIMAX_H3_CP_PLAN,
            )

            for block in self.transformer.token_refiner.refiner_blocks:
                block.attn.processor._parallel_config = None

        from . import adaln

        group = self._get_parallel_mesh("cp")["ulysses"].get_group() if self.ulysses_degree > 1 else None
        self._install_transformer(self.transformer, group=group)
        if self.runtime.adaln_precompute:
            adaln.enable_adaln_precompute(
                self.transformer,
                verbose=self.rank == 0,
                component_name="transformer_ref" if task == "ref2va" else "transformer",
            )
        self._install_vae(self.pipe.vae, encode_parallel=task == "ref2va")
        self._adaln_checked = False

    def _get_parallel_mesh(self, kind):
        """Own each mesh for the engine lifetime, across staged DiT reloads."""
        from torch.distributed.device_mesh import init_device_mesh

        if self._closed:
            raise RuntimeError("The inference engine is closed")
        if kind == "fsdp" and self.runtime.fsdp_degree in (None, 1):
            return None
        if kind not in self._meshes:
            if kind == "cp":
                if self.ulysses_degree == 1:
                    # LTX retains its native SP wrapper with one member per group.
                    mesh = init_device_mesh("cuda", (self.world_size, 1),
                                            mesh_dim_names=("replica", "ulysses"))
                else:
                    mesh = init_device_mesh("cuda", (1, self.ulysses_degree), mesh_dim_names=("ring", "ulysses"))
            elif kind == "fsdp":
                mesh = init_device_mesh("cuda", (self.runtime.fsdp_degree,), mesh_dim_names=("fsdp",))
            else:
                raise ValueError(f"Unknown parallel mesh: {kind}")
            self._meshes[kind] = mesh
            groups = mesh.get_all_groups()
            if kind == "cp" and self.ulysses_degree > 1:
                # Diffusers also uses this flattened view; create and track it once.
                groups += mesh["ring", "ulysses"]._flatten().get_all_groups()
            for group in groups:
                if group is not dist.group.WORLD and all(group is not owned for owned in self._mesh_groups):
                    self._mesh_groups.append(group)
        return self._meshes[kind]

    def _install_vae(self, vae, *, encode_parallel=False):
        """Use the same fallback settings for resident, conditioning and decode VAEs."""
        if self.runtime.vae_tile_size is not None:
            vae.enable_tiling(tile_sample_min_height=self.runtime.vae_tile_size,
                              tile_sample_min_width=self.runtime.vae_tile_size,
                              tile_sample_min_overlap_height=self.runtime.vae_tile_size//4,
                              tile_sample_min_overlap_width=self.runtime.vae_tile_size//4)
        if not self.runtime.parallel_vae:
            return lambda: None
        from . import vae_parallel

        mode = self.runtime.vae_compile_mode
        if mode is None:
            mode = os.environ.get("H3_VAE_COMPILE", "default")
        if mode.lower() in {"", "0", "off", "false", "eager"}:
            mode = ""
        return vae_parallel.install(
            vae, batched=self.runtime.vae_batched,
            global_batch=self.runtime.vae_global_batch, compile_mode=mode,
            encode_parallel=encode_parallel,
        )

    def _install_transformer(self, transformer, group) -> None:
        """Install the official optional kernels before the FSDP parameter layout is frozen."""
        from . import fusion_install, sparse_attention, ulysses

        compute_quant, lora_mode = self.compute_quant, self.lora_mode
        attention_backend, task = self.attention_backend, self.task
        self.compute_quant_report = None
        if compute_quant == "mxfp8":
            from .compute_quant import install as install_compute_quant

            self.compute_quant_report = install_compute_quant(transformer)
            if self.rank == 0:
                report = self.compute_quant_report
                ratio = report.original_bytes / report.quantized_bytes
                print(
                    "MXFP8 compute: "
                    f"blocks={report.first_block}..{report.last_block}, "
                    f"linears={report.quantized_linears}, "
                    f"storage={report.original_bytes / 2**30:.2f}->"
                    f"{report.quantized_bytes / 2**30:.2f} GiB ({ratio:.2f}x)"
                )
        if lora_mode == "fused":
            from .lora_fusion import install as install_lora_fusion

            self.lora_fusion = install_lora_fusion(transformer)
        if self.runtime.kernel_fusions:
            fusion_install.install(transformer, lora=lora_mode == "fused")
        self.sparse_attention = None
        if attention_backend != "dense":
            self.sparse_attention = sparse_attention.install(
                transformer,
                backend=attention_backend,
                tau=1.0,
                dense_steps=0 if task == "ref2va" else 1,
                dense_layers=0 if task == "ref2va" else 2,
                sink_mode="text_audio" if task == "ref2va" else "prefix",
            )
        if self.ulysses_degree > 1 and self.runtime.packed_sp:
            ulysses.install(
                transformer,
                group=group,
                attention_fn=self.sparse_attention,
                lora=lora_mode == "fused",
            )
        if self.runtime.flash_sdpa:
            # Restrict only DiT attention; FP32 audio reference encoders need other kernels.
            transformer.forward = sdpa_kernel(SDPBackend.FLASH_ATTENTION)(transformer.forward)

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def _check_adaln(self) -> None:
        if not self.runtime.adaln_precompute:
            return
        if self._adaln_checked:
            return
        for index, block in enumerate(self.transformer.transformer_blocks):
            table = getattr(block.adaln_proj, "table", None)
            schedules = 3 if self.task == "ref2va" else 2
            if table is None or table.ndim != 4 or table.shape[0] != schedules:
                raise RuntimeError(f"invalid AdaLN cache at transformer block {index}")
        self._adaln_checked = True

    @staticmethod
    def _load_image(image: str | Path | Image.Image | None) -> Image.Image | None:
        if image is None:
            return None
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image).convert("RGB")
        with Image.open(image) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        duration: int,
        seed: int = 0,
        image: str | Path | Image.Image | None = None,
        references: list[Any] | None = None,
        width: int = WIDTH,
        height: int = HEIGHT,
        capture_latents: bool = False,
        num_inference_steps: int = INFERENCE_STEPS,
    ) -> GeneratedMedia | GeneratedLatents | None:
        """Generate one video. ``references`` is required only for Ref2VA."""
        if getattr(self, "_closed", False):
            raise RuntimeError("The inference engine is closed")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if duration not in DURATION_FRAMES:
            raise ValueError("duration must be 5, 10, or 15 seconds")
        if type(num_inference_steps) is not int or num_inference_steps < 2:
            raise ValueError("num_inference_steps must contain at least two scheduler points")
        if num_inference_steps != INFERENCE_STEPS and self.runtime.adaln_precompute:
            raise ValueError("Disable AdaLN precompute when changing the released four-update schedule")
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError("width and height must be positive multiples of 32")
        if capture_latents and (not self.runtime.staged_loading or self.task != "t2v"):
            raise ValueError("normalized latent capture requires staged T2V")
        if self.task == "ref2va":
            if image is not None:
                raise ValueError("image is a first-frame I2V input; use references for Ref2VA")
            if not references:
                raise ValueError("Ref2VA requires at least one reference")
        elif references is not None:
            raise ValueError("references are only valid when task='ref2va'")

        first_frame = self._load_image(image)
        if self.world_size > 1:
            from .ulysses import reset_row_counts

            reset_row_counts()
            dist.barrier()

        request = {
            "prompt": prompt.strip(),
            "height": height,
            "width": width,
            "num_frames": DURATION_FRAMES[duration],
            "num_inference_steps": num_inference_steps,
            "generator": torch.Generator().manual_seed(int(seed)),
            "output_type": "pt",
        }
        if first_frame is not None:
            request["image"] = first_frame
        if self.task == "ref2va":
            request["references"] = list(references)

        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        # FSDP must materialize ordinary tensors; retain the upstream inference mode otherwise.
        self._capture_latents = capture_latents
        try:
            with torch.inference_mode(not self.runtime.staged_loading):
                state = self.pipe(**request)
        finally:
            self._capture_latents = False
        torch.cuda.synchronize(self.device)
        if self.world_size > 1:
            dist.barrier()
        elapsed = time.perf_counter() - started
        self._check_adaln()

        if not self.is_rank_zero:
            return None
        if capture_latents:
            return GeneratedLatents(
                video=state.values["h3_normalized_native"], audio=_state_value(state, "audio"),
                audio_sample_rate=_state_value(state, "sampling_rate"), elapsed_s=elapsed, seed=int(seed),
            )
        videos = _state_value(state, "videos")
        if videos is None:
            raise RuntimeError("pipeline returned no video")
        audio = _state_value(state, "audio")
        return GeneratedMedia(
            video=videos[0],
            audio=None if audio is None else audio[0],
            audio_sample_rate=_state_value(state, "sampling_rate"),
            elapsed_s=elapsed,
            duration=duration,
            seed=int(seed),
        )

    def generate_to_file(
        self,
        output_path: str | Path,
        prompt: str,
        *,
        duration: int,
        seed: int = 0,
        image: str | Path | Image.Image | None = None,
        references: list[Any] | None = None,
    ):
        """Generate and start background MP4 encoding on rank zero."""
        media = self.generate(
            prompt,
            duration=duration,
            seed=seed,
            image=image,
            references=references,
        )
        return None if media is None else media.save_async(output_path)

    def warmup(
        self,
        *,
        duration: int = 5,
        prompt: str | None = None,
        image: str | Path | Image.Image | None = None,
        references: list[Any] | None = None,
    ) -> None:
        self.generate(
            prompt
            or "A continuous cinematic shot of morning light moving across a quiet room.",
            duration=duration,
            seed=0,
            image=image,
            references=references,
        )

    def close(self, *, synchronize: bool = True) -> None:
        if getattr(self, "_closed", False):
            return
        stages = getattr(self, "_stages", None)
        if stages is not None:
            stages.close()
        elif getattr(self, "cpu_group", None) is not None and dist.is_initialized():
            # initialize() may fail after creating this group, before Stages exists.
            dist.destroy_process_group(self.cpu_group)
        for group in reversed(getattr(self, "_mesh_groups", [])):
            if dist.is_initialized():
                dist.destroy_process_group(group)
        self._mesh_groups = []
        self._meshes = {}
        if self._owns_process_group and dist.is_initialized():
            if synchronize and stages is None:
                dist.barrier()
            dist.destroy_process_group()
            self._owns_process_group = False
        self.transformer = self.pipe = self.lora_fusion = self.sparse_attention = None
        self._stages = None
        gc.collect()
        torch.cuda.empty_cache()
        self._closed = True

    def __enter__(self) -> "MiniMaxH3Inference":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(synchronize=exc_type is None)
