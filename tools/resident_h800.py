"""Persistent single-H800 FastH3 backend; upstream and frozen sources stay intact."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import uuid

_ENGINE_INIT_LOCK = threading.Lock()


class H800ResidentBackend:
    def __init__(self, model, adapter):
        started = time.monotonic()
        for name, value in {"DISABLE_ADDMM_CUDA_LT": "1", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "TOKENIZERS_PARALLELISM": "false"}.items():
            os.environ.setdefault(name, value)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters/rtx4090"))
        import torch
        import torch.distributed as dist
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
        from diffusers.hooks import apply_group_offloading
        from h3_runtime import staged
        from h3_runtime.engine import MiniMaxH3Inference, RuntimeConfig
        from h3_runtime.fsdp import load_transformer

        self.engine = None
        self.rendezvous = None
        self.owns_group = False
        self.fatal_error = None
        self.stats = dict(backend="h800-resident", worker_pid=os.getpid(),
                          model_instance_id=uuid.uuid4().hex, requests_completed=0,
                          placement="DiT and VAEs on GPU; text encoder retained on CPU with leaf offload")
        self.stages = None
        torch.cuda.set_device(0)
        assert torch.cuda.device_count() == 1, "Resident H800 profile requires exactly one visible GPU"
        props = torch.cuda.get_device_properties(0)
        assert "H800" in props.name and (props.major, props.minor) == (9, 0), props.name
        assert props.total_memory >= 78 * 2**30, "Resident H800 profile requires 80 GiB"
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
        owner = self

        class ResidentStages(staged.Stages):
            def mark(self, stage):
                record = json.dumps(dict(stage=stage, worker_pid=os.getpid()), ensure_ascii=False)
                print(record, flush=True)
                if getattr(owner, "request_log", None) is not None:
                    owner.request_log.write((record + "\n").encode())
                    owner.request_log.flush()

            def text(self, original, block, components, state):
                self.mark("text_encode_resident")
                return original(block, components, state)

            def denoise(self, original, block, components, state):
                if not torch.equal(state.timesteps, owner.video_times) or not torch.equal(
                        state.audio_timesteps, owner.audio_times):
                    raise RuntimeError("Request schedule differs from the resident AdaLN tables")
                self.mark("denoise_resident")
                return original(block, components, state)

            def condition(self, original, block, components, state):
                return original(block, components, state)

            def video(self, original, block, components, state):
                self.mark("video_decode_resident")
                return original(block, components, state)

            def audio(self, original, block, components, state):
                self.mark("audio_decode_resident")
                components, state = original(block, components, state)
                state.values["audio"] = staged._cpu_copy(state.audio)
                return components, state

        try:
            if not dist.is_initialized():
                self.rendezvous = tempfile.TemporaryDirectory(prefix="sol-h3-resident-dist-")
                dist.init_process_group("nccl", rank=0, world_size=1,
                    init_method=(Path(self.rendezvous.name) / "rendezvous").as_uri(),
                    device_id=torch.device("cuda", 0))
                self.owns_group = True
            assert dist.get_world_size() == 1, "This resident backend is single GPU"
            runtime = RuntimeConfig(fsdp_degree=1, ulysses_degree=1, staged_loading=True,
                flash_sdpa=True, allow_tf32=False, allow_bf16_reduced_precision_reduction=False,
                packed_sp=False, parallel_vae=False, vae_compile_mode="eager", vae_tile_size=256)
            # Select the component lifetime policy while the frozen initializer installs its hooks.
            # The symbol is restored immediately; no bundled source file is modified.
            with _ENGINE_INIT_LOCK:
                original_stages = staged.Stages
                staged.Stages = ResidentStages
                try:
                    self.engine = MiniMaxH3Inference(str(model), adapter,
                        attention_backend="dense", task="t2v", runtime=runtime)
                finally:
                    staged.Stages = original_stages
            self.stages = self.engine._stages
            pipe = self.engine.pipe
            pipe.scheduler.set_timesteps(5, device=self.engine.device)
            pipe.audio_scheduler.set_timesteps(5, device=self.engine.device)
            self.video_times = pipe.scheduler.timesteps.clone()
            self.audio_times = pipe.audio_scheduler.timesteps.clone()
            self.stages.mark("preload_dit")
            transformer = load_transformer(Path(model), Path(adapter) if adapter else None,
                self.engine.cpu_group, self.video_times, self.audio_times, engine=self.engine)
            pipe.register_components(transformer=transformer)
            self.engine.transformer = transformer
            self.engine._check_adaln()
            self.stages.mark("preload_text_encoder")
            pipe.load_components(names=["text_encoder", "tokenizer", "processor"],
                dtype=torch.bfloat16, pretrained_model_name_or_path=str(model))
            pipe.text_encoder.eval().requires_grad_(False)
            apply_group_offloading(pipe.text_encoder, onload_device=self.engine.device,
                                   offload_type="leaf_level", use_stream=False)
            self.stages.mark("preload_video_vae")
            vae = AutoencoderKLMiniMaxH3.from_pretrained(str(Path(model) / "vae"),
                torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False)
            vae.to(self.engine.device)
            vae.enable_tiling()
            self.engine._install_vae(vae)
            self.stages.mark("preload_audio_vae")
            audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(Path(model) / "audio_vae"),
                torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False)
            audio_vae.to(self.engine.device)
            pipe.register_components(vae=vae, audio_vae=audio_vae)
            self.component_ids = {name: id(getattr(pipe, name)) for name in
                                  ("transformer", "text_encoder", "vae", "audio_vae")}
            self.dit_pointer = next(transformer.parameters()).data_ptr()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.stats.update(model_load_count=1, component_load_counts={n: 1 for n in self.component_ids},
                component_ids=self.component_ids, startup_s=time.monotonic() - started,
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved())
            self.stages.mark("resident_ready")
        except BaseException:
            self.close()
            raise

    def generate(self, parameters, output, log):
        import torch
        started = time.monotonic()
        self.request_log = log
        media = None
        try:
            torch.cuda.reset_peak_memory_stats()
            self._check_identity()
            log.write((json.dumps(dict(event="resident_request_start", **self.stats)) + "\n").encode())
            log.flush()
            media = self.engine.generate(parameters["prompt"], duration=parameters["duration"],
                                         seed=parameters["seed"])
            generation_s = media.elapsed_s
            encode_started = time.monotonic()
            media.save(output)
            encoding_s = time.monotonic() - encode_started
            media = None
            gc.collect()
            torch.cuda.synchronize()
            self._check_identity()
            self.stats["requests_completed"] += 1
            self.stats.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                              cuda_reserved_bytes=torch.cuda.memory_reserved())
            result = dict(worker_pid=os.getpid(), model_instance_id=self.stats["model_instance_id"],
                model_load_count=1, request_index=self.stats["requests_completed"],
                component_ids=dict(self.component_ids),
                generation_s=generation_s, encoding_s=encoding_s,
                request_wall_s=time.monotonic() - started,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            log.write((json.dumps(dict(event="resident_request_complete", **result)) + "\n").encode())
            log.flush()
            return result
        except BaseException as exc:
            # A failed CUDA request may leave scheduler/hooks in an unknown state. Preserve
            # the queue and report unready rather than silently rebuilding weights per job.
            self.fatal_error = f"Resident request failed: {type(exc).__name__}: {exc}"
            raise
        finally:
            media = None
            self.request_log = None

    def _check_identity(self):
        for name, identity in self.component_ids.items():
            if id(getattr(self.engine.pipe, name)) != identity:
                raise RuntimeError(f"Resident component was replaced: {name}")
        if next(self.engine.transformer.parameters()).data_ptr() != self.dit_pointer:
            raise RuntimeError("Resident DiT weight storage changed")

    def health(self):
        return dict(self.stats)

    def close(self):
        import torch.distributed as dist
        if self.engine is not None:
            self.engine.close(synchronize=False)
            self.engine = None
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()
            self.owns_group = False
        if self.rendezvous is not None:
            self.rendezvous.cleanup()
            self.rendezvous = None
