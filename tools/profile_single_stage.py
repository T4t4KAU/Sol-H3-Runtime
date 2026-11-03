#!/usr/bin/env python3
"""Observe the existing staged four-forward adapter without editing frozen sources.

Launch once per rank with torchrun. Pass normal infer.py arguments after --.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/rtx4090"


def indices(value):
    try:
        result = tuple(int(x) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integer indices") from exc
    if not result or len(set(result)) != len(result) or any(x < 0 for x in result):
        raise argparse.ArgumentTypeError("Indices must be distinct and nonnegative")
    return tuple(sorted(result))


def configure(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--capture", choices=("profile", "benchmark"), default="profile")
    parser.add_argument("--forward-indices", type=indices, default=(0, 2, 3))
    parser.add_argument("--profile-ranks", type=indices, default=(0,))
    parser.add_argument("--check-arguments", action="store_true", help="Parse only; no torch import or GPU execution")
    parser.add_argument("inference_args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    inference_args = options.inference_args
    if inference_args[:1] == ["--"]:
        inference_args = inference_args[1:]
    if any(i >= 4 for i in options.forward_indices):
        parser.error("Four-step forward indices must be 0, 1, 2, or 3")
    sys.path.insert(0, str(ADAPTER))
    spec = importlib.util.spec_from_file_location("sol_h3_profile_infer", ADAPTER / "infer.py")
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)
    resolved = infer.parse_args(inference_args)
    if resolved.mode != "single-stage" or not resolved.staged_loading or resolved.warmup:
        parser.error("Require --mode single-stage --staged-loading without --warmup")
    if any(getattr(resolved, name, None) is not None for name in
           ("profile_dir", "benchmark_dir", "evidence_dir", "raw_output_path")):
        parser.error("Use wrapper --artifacts; do not combine with other capture flags")
    if not options.check_arguments and max(options.profile_ranks) >= int(os.environ.get("WORLD_SIZE", "1")):
        parser.error("A requested profiling rank is outside WORLD_SIZE")
    return options, inference_args, infer, resolved


def main(argv=None):
    options, inference_args, infer, resolved = configure(argv)
    if options.check_arguments:
        print(json.dumps(dict(status="arguments_valid", cuda_executed=False,
                              mode=resolved.mode, expected_forwards=4,
                              capture=options.capture, forward_indices=options.forward_indices,
                              profile_ranks=options.profile_ranks), indent=2))
        return 0

    import torch
    import torch.distributed as dist
    from h3_runtime import MiniMaxH3Inference
    from h3_runtime.profiling import RunInstrumentation, source_hashes, write_json

    rank = int(os.environ.get("RANK", "0"))
    directory = options.artifacts.expanduser().resolve() / f"rank{rank}"
    directory.mkdir(parents=True, exist_ok=False)
    profile = options.capture == "profile" and rank in options.profile_ranks
    metadata = dict(rank=rank, world_size=int(os.environ.get("WORLD_SIZE", "1")),
                    requested_capture=options.capture, cuda_trace_enabled=profile,
                    expected_forwards=4, selected_forward_indices=options.forward_indices if profile else [],
                    inference_args=inference_args,
                    source_hashes=source_hashes(ADAPTER),
                    wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    command=sys.argv, torch=torch.__version__, cuda_runtime=torch.version.cuda,
                    environment={key: os.environ.get(key) for key in
                                 ("CUDA_VISIBLE_DEVICES", "DISABLE_ADDMM_CUDA_LT", "NCCL_P2P_DISABLE",
                                  "NCCL_SHM_DISABLE", "OMP_NUM_THREADS")},
                    scope="Observe four-forward staged adapter; no arithmetic or memory-policy changes")
    write_json(directory / "metadata.json", metadata)
    original_generate = MiniMaxH3Inference.generate
    calls = 0

    @functools.wraps(original_generate)
    def generate(engine, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls != 1 or engine.instrumentation is not None:
            raise RuntimeError("One request per run; do not combine observers")
        if not engine.runtime.staged_loading:
            raise RuntimeError("This observer requires staged loading")
        recorder = RunInstrumentation(
            directory, profile=profile, rank=rank, device=engine.device, torch_module=torch,
            indices=options.forward_indices, expected_forwards=4, label="single_stage4",
            flash_sdpa=resolved.flash_sdpa, attention_backend=resolved.attention_backend)
        recorder.cuda_peak_scope = (
            "Since observer setup before generate(); includes request component loading and decoding; "
            "excludes engine construction and MP4 encoding")
        original_phase = recorder.phase

        @contextmanager
        def measured_phase(name):
            if options.capture != "benchmark" or name != "denoise":
                with original_phase(name):
                    yield
                return
            # Time only the staged denoise body, after initial weight loading and
            # before transformer release/VAE loading. Includes FSDP/SP work.
            if dist.is_initialized():
                dist.barrier()
            torch.cuda.synchronize(engine.device)
            begin = time.perf_counter()
            try:
                with original_phase(name):
                    yield
            finally:
                torch.cuda.synchronize(engine.device)
                recorder.timings["synchronized_denoise_s"] = time.perf_counter() - begin

        recorder.phase = measured_phase
        engine.instrumentation = recorder
        error = None
        started = None
        try:
            if dist.is_initialized():
                dist.barrier()
            torch.cuda.synchronize(engine.device)
            started = time.perf_counter()
            media = original_generate(engine, *args, **kwargs)
            torch.cuda.synchronize(engine.device)
            recorder.timings["synchronized_generate_s"] = time.perf_counter() - started
            if options.capture == "benchmark" and "synchronized_denoise_s" not in recorder.timings:
                raise RuntimeError("The staged denoise timing boundary was not entered")
            recorder.success = True
            return media
        except BaseException as exc:
            error = exc
            raise
        finally:
            if started is not None and error is not None:
                recorder.timings["failed_generate_observation_s"] = time.perf_counter() - started
            try:
                recorder.finish(error)
            finally:
                engine.instrumentation = None

    MiniMaxH3Inference.generate = generate
    started = time.time()
    try:
        result = infer.main(inference_args)
        if calls != 1:
            raise RuntimeError(f"Expected one generate call, got {calls}")
        write_json(directory / "launch-result.json", dict(
            status="completed", exit_code=result or 0, started_epoch=started, finished_epoch=time.time(),
            cuda_trace_enabled=profile, expected_forwards=4,
            timing_scope="See summary-rankN.json; generate timing excludes engine construction and MP4 encoding."
        ))
        return result or 0
    except BaseException as exc:
        write_json(directory / "launch-result.json", dict(
            status="failed", started_epoch=started, finished_epoch=time.time(),
            error=str(exc), traceback=traceback.format_exc(), generate_calls=calls))
        raise
    finally:
        MiniMaxH3Inference.generate = original_generate


if __name__ == "__main__":
    raise SystemExit(main())
