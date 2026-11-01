#!/usr/bin/env bash
# Default entry: the unchanged official SOL-H3 implementation.
set -euo pipefail
SOL_H3_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SOL_H3_PACKAGE/run_official.sh" "$@"
