#!/usr/bin/env bash
# Keep official and adapted h3_runtime modules in separate interpreter paths.
set -euo pipefail
SOL_H3_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOL_H3_OFFICIAL_PYTHON="${SOL_H3_OFFICIAL_PYTHON:-python}"
SOL_H3_OFFICIAL_GPUS="${SOL_H3_OFFICIAL_GPUS:-8}"
for arg in "$@"; do
    if [[ "$arg" == --help || "$arg" == -h ]]; then
        cat <<'USAGE'
Run the unchanged upstream Sol-H3 infer.py.
Environment: SOL_H3_OFFICIAL_PYTHON (default python), SOL_H3_OFFICIAL_GPUS (1/2/4/8, default 8).
Arguments: --model DIR --adapter FILE --task t2v --duration 5 --prompt TEXT --output FILE
Optional upstream arguments: --attention-backend dense|sol|sol_bsa --compute-quant none|mxfp8
Use official/Sol-H3/infer.py --help in the official Python environment for all options.
This is the resident upstream entry; SP does not provide FSDP weight sharding.
Server-specific deployment configurations are listed in docs/LINS_LAB.md.
USAGE
        exit 0
    fi
done
case "$SOL_H3_OFFICIAL_GPUS" in
    1) exec "$SOL_H3_OFFICIAL_PYTHON" "$SOL_H3_PACKAGE/official/Sol-H3/infer.py" "$@" ;;
    2|4|8) exec "$SOL_H3_OFFICIAL_PYTHON" -m torch.distributed.run --standalone \
        "--nproc-per-node=$SOL_H3_OFFICIAL_GPUS" "$SOL_H3_PACKAGE/official/Sol-H3/infer.py" "$@" ;;
    *) echo 'SOL_H3_OFFICIAL_GPUS must be 1, 2, 4, or 8' >&2; exit 2 ;;
esac
