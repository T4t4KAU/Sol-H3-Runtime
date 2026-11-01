#!/usr/bin/env python3
"""Verify bundled upstream and frozen adaptation files without importing torch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify(root: Path) -> dict:
    manifest = json.loads((root / "SOURCES.json").read_text())
    checked = {}
    for profile in ("official", "rtx4090"):
        source = manifest[profile]
        directory = root / source["destination"]
        count = 0
        for name, entry in source["files"].items():
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Missing regular source file: {profile}/{name}")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"Source changed: {profile}/{name}")
            if "git_blob_sha1" in entry:
                blob = f"blob {len(data)}\0".encode() + data
                if hashlib.sha1(blob).hexdigest() != entry["git_blob_sha1"]:
                    raise ValueError(f"Upstream Git blob differs: {name}")
            count += 1
        checked[profile] = count
    return {"status": "passed", "files_verified": checked,
            "official_revision": manifest["official"]["revision"],
            "cuda_executed": False}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
