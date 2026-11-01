#!/usr/bin/env bash
# Complete-generation profile recorded on LINs Lab RTX4090 24 GiB nodes.
set -euo pipefail
SOL_H3_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SOL_H3_PACKAGE/configs/lins_rtx4090_24g_8gpu.env"
export SOL_H3_PYTHON="${SOL_H3_PYTHON:-$SOL_H3_PROFILE_PYTHON}"
export SOL_H3_ASSETS="${SOL_H3_ASSETS:-$SOL_H3_PROFILE_ASSETS}"
export SOL_H3_FSDP_DEGREE="$SOL_H3_PROFILE_FSDP_DEGREE"
exec bash "$SOL_H3_PACKAGE/run_8x4090.sh" "$@"
