#!/bin/bash
set -e

MAX_THREADS="${1:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"

apptainer exec --cleanenv --no-home \
    --bind "$REPO_ROOT:/repo" \
    --env HOME=/root \
    --env MAX_THREADS="$MAX_THREADS" \
    "$SIF_IMAGE" bash -c '
cd /repo/mttkrp
if [ ! -d data ] || [ -z "$(ls -A data 2>/dev/null)" ]; then
    bash get_mttkrp_data.sh
fi
t=1
while [ "$t" -le "$MAX_THREADS" ]; do
    export OMP_NUM_THREADS="$t"
    export MKL_NUM_THREADS="$t"
    julia -t "$t" run_mttkrp.jl --dataset sparse --output "results/mttkrp_sparse_t${t}.json" --ncpu "$t"
    t=$((t * 2))
done
cd results
poetry run python3 plot_mttkrp_strong_scaling.py
'
