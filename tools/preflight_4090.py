#!/usr/bin/env python3
"""Check source identity, assets, and optionally the actual eight-GPU environment."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import sys

from verify_sources import verify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--gpu", action="store_true", help="Run CUDA checks on the allocated GPU server")
    parser.add_argument("--launcher-args", nargs=argparse.REMAINDER,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_integrity = verify(root)
    if args.launcher_args is not None:
        # Reuse the frozen CLI to preserve path discovery and argument handling.
        spec = importlib.util.spec_from_file_location(
            "sol_h3_adapter_cli", root / "adapters/rtx4090/infer.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        launch = cli.parse_args(["--no-adaln-precompute", *args.launcher_args,
                                 "--mode", "single-stage"])
        args.model, args.adapter = launch.model, launch.adapter
    if args.model is None or args.adapter is None:
        parser.error("Provide --model and --adapter")
    result = {"source_integrity": source_integrity, "errors": [], "warnings": []}
    errors = result["errors"]
    model = args.model.expanduser().resolve()
    adapter = args.adapter.expanduser().resolve()
    required = ["transformer", "text_encoder", "tokenizer",
                "processor", "scheduler", "audio_scheduler", "vae", "audio_vae"]
    missing = [name for name in required if not (model / name).exists()]
    if not any((model / name).is_file() for name in ("model_index.json", "modular_model_index.json")):
        missing.append("model_index.json or modular_model_index.json")
    if missing:
        errors.append({"missing_model_components": missing, "model": str(model)})
    if not adapter.is_file():
        errors.append({"missing_adapter": str(adapter)})
    versions = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "av"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append({"missing_dependency": name})
    result["versions"] = versions
    result["python"] = sys.executable
    result["media_tools"] = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    if not result["media_tools"]["ffprobe"]:
        result["warnings"].append("ffprobe is unavailable; install it to inspect generated media")
    # The runtime uses PyAV for encoding; ffmpeg CLI is not a mandatory runtime dependency here.
    if "torch" in versions and not versions["torch"].startswith("2.11.0"):
        result["warnings"].append("The frozen 4090 baseline used torch2.11.0+cu128")
    if args.gpu:
        if importlib.util.find_spec("torch") is None:
            errors.append("GPU check requires PyTorch in this interpreter")
        else:
            import torch
            result["cuda_runtime"] = torch.version.cuda
            devices = []
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                item = {"index": i, "name": prop.name, "capability": [prop.major, prop.minor],
                        "memory_gib": prop.total_memory / 2**30}
                devices.append(item)
                if "4090" not in prop.name or (prop.major, prop.minor) != (8, 9):
                    errors.append({"unexpected_device": item})
                if not 22 <= item["memory_gib"] <= 26:
                    errors.append({"expected_24GiB_device": item})
            result["devices"] = devices
            if len(devices) != 8:
                errors.append("Eight visible RTX4090 24GiB GPUs are required")
    result["status"] = "passed" if not errors else "failed"
    result["scope"] = "Startup check only; does not load weights or prove full generation correctness"
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
