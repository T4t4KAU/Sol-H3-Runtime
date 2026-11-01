#!/usr/bin/env bash
# Optional H3+LTX path; additional external sources/assets are required.
set -euo pipefail
SOL_H3_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SOL_H3_PACKAGE/adapters/rtx4090/run_two_stage_8x4090.sh" "$@"
