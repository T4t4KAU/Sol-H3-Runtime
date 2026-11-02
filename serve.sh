#!/usr/bin/env bash
# Start the HTTP inference service on the GPU server.
set -euo pipefail
SOL_H3_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOL_H3_API_PYTHON="${SOL_H3_API_PYTHON:-${SOL_H3_PYTHON:-python3}}"
exec "$SOL_H3_API_PYTHON" "$SOL_H3_PACKAGE/tools/serve_api.py" "$@"
