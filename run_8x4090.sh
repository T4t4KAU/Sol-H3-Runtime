#!/usr/bin/env bash
# Optional adapter: the historically validated 8xRTX4090 24 GiB profile.
set -euo pipefail
SOL_H3_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SOL_H3_PACKAGE/adapters/rtx4090/run_single_stage_8x4090.sh" "$@"
