#!/usr/bin/env bash
# One H800: stage components to keep the complete pipeline within 80 GiB.
set -euo pipefail
SOL_H3_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOL_H3_PYTHON="${SOL_H3_PYTHON:-python}"
for arg in "$@"; do
    if [[ "$arg" == --help || "$arg" == -h ]]; then
        exec "$SOL_H3_PYTHON" "$SOL_H3_REPO/adapters/rtx4090/infer.py" --help
    fi
done
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 DISABLE_ADDMM_CUDA_LT=1
"$SOL_H3_PYTHON" - <<'CHECK'
import torch
assert torch.cuda.is_available(), "A Slurm GPU allocation is required"
assert torch.cuda.device_count() == 1, "Exactly one visible GPU is required"
p = torch.cuda.get_device_properties(0)
assert "H800" in p.name and (p.major, p.minor) == (9, 0), p.name
assert p.total_memory >= 78 * 2**30, "An H800 with approximately 80 GiB is required"
CHECK
exec "$SOL_H3_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=1 \
    "$SOL_H3_REPO/adapters/rtx4090/infer.py" \
    --fsdp-degree 1 --ulysses-degree 1 --staged-loading --flash-sdpa \
    --no-tf32 --no-bf16-reduced-precision-reduction \
    --no-packed-sp --no-parallel-vae --vae-compile-mode eager --vae-tile-size 256 \
    --attention-backend dense --compute-quant none --lora-mode merged \
    "$@" --mode single-stage
