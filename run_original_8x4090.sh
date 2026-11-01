#!/usr/bin/env bash
set -euo pipefail
SOL_H3_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SOL_H3_PACKAGE/adapters/rtx4090/run_original_8x4090.sh" "$@"
