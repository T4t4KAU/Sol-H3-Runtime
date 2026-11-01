#!/usr/bin/env bash
# Eight RTX 4090s. Keep weights, environments and generated files outside this repo.
set -euo pipefail
SOL_H3_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOL_H3_PYTHON="${SOL_H3_PYTHON:-python}"
for arg in "$@"; do
    if [[ "$arg" == --help || "$arg" == -h ]]; then
        exec "$SOL_H3_PYTHON" "$SOL_H3_SOURCE/infer.py" --help
    fi
done
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Preserve the accepted original-H3 numerical setting; LTX uses a different default.
export DISABLE_ADDMM_CUDA_LT="${DISABLE_ADDMM_CUDA_LT:-0}"
"$SOL_H3_PYTHON" - <<'CHECK'
import torch
if torch.cuda.device_count() != 8:
    raise SystemExit('Eight visible RTX 4090 GPUs are required')
for index in range(8):
    gpu = torch.cuda.get_device_properties(index)
    if '4090' not in gpu.name or (gpu.major, gpu.minor) != (8, 9):
        raise SystemExit(f'Expected RTX 4090 (SM89), got {gpu.name}')
CHECK
# Without AdaLN precompute, H3 weights exceed 48GB. Replication is an explicit override.
exec "$SOL_H3_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 \
    "$SOL_H3_SOURCE/infer.py" --fsdp-degree "${SOL_H3_FSDP_DEGREE:-8}" \
    --ulysses-degree 8 --staged-loading --text-fsdp --flash-sdpa \
    --no-tf32 --no-bf16-reduced-precision-reduction --no-adaln-precompute \
    --no-kernel-fusions --no-packed-sp --no-parallel-vae --vae-compile-mode eager \
    --vae-tile-size 256 --attention-backend dense --compute-quant none --lora-mode merged \
    "$@" --mode original
