#!/usr/bin/env bash
# Eight RTX 4090s by default; explicit four-card runs are experimental.
# Keep weights, environments and generated files outside this repo.
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
export DISABLE_ADDMM_CUDA_LT="${DISABLE_ADDMM_CUDA_LT:-1}"
export SOL_H3_GPU_COUNT="${SOL_H3_GPU_COUNT:-8}"
if [[ -z "${SOL_H3_RANK_LAUNCHER:-}" ]]; then
"$SOL_H3_PYTHON" - <<'CHECK'
import os
import torch
count = int(os.environ['SOL_H3_GPU_COUNT'])
if count not in (4, 8) or torch.cuda.device_count() != count:
    raise SystemExit('Exactly four or eight visible RTX 4090 GPUs are required; four is experimental')
for index in range(count):
    gpu = torch.cuda.get_device_properties(index)
    if '4090' not in gpu.name or (gpu.major, gpu.minor) != (8, 9):
        raise SystemExit(f'Expected RTX 4090 (SM89), got {gpu.name}')
CHECK
fi
# Without AdaLN precompute, H3 weights exceed 48GB. Replication is an explicit override.
if [[ -n "${SOL_H3_RANK_LAUNCHER:-}" ]]; then
    SOL_H3_LAUNCH=("$SOL_H3_PYTHON" "$SOL_H3_RANK_LAUNCHER" --check-4090 --nproc-per-node="$SOL_H3_GPU_COUNT")
else
    SOL_H3_LAUNCH=("$SOL_H3_PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$SOL_H3_GPU_COUNT")
fi
exec "${SOL_H3_LAUNCH[@]}" \
    "$SOL_H3_SOURCE/infer.py" --fsdp-degree "${SOL_H3_FSDP_DEGREE:-$SOL_H3_GPU_COUNT}" \
    --ulysses-degree "$SOL_H3_GPU_COUNT" --staged-loading --text-fsdp --flash-sdpa \
    --no-tf32 --no-bf16-reduced-precision-reduction --no-adaln-precompute \
    --no-kernel-fusions --no-packed-sp --no-parallel-vae --vae-compile-mode eager \
    --vae-tile-size 256 --attention-backend dense --compute-quant none --lora-mode merged \
    --ltx-attention-backend efficient --ltx-linear-mode full-sequence \
    "$@" --mode two-stage
