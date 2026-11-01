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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true", help="Run CUDA checks on the allocated GPU server")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = {"source_integrity": verify(root), "errors": [], "warnings": []}
    errors = result["errors"]
    model = args.model.expanduser().resolve()
    adapter = args.adapter.expanduser().resolve()
    required = ["model_index.json", "transformer", "text_encoder", "tokenizer",
                "processor", "scheduler", "audio_scheduler", "vae", "audio_vae"]
    missing = [name for name in required if not (model / name).exists()]
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
