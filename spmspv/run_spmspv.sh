#!/bin/bash
set -e

THREADS="${1:-16}"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"

apptainer exec --cleanenv --no-home \
    --bind "$REPO_ROOT:/repo" \
    --env OMP_NUM_THREADS="$THREADS" \
    --env MKL_NUM_THREADS="$THREADS" \
    "$SIF_IMAGE" bash -c "
cd /repo/spmspv
julia -t $THREADS run_spmspv.jl --dataset snap_largest --output results/spmspv_results.json
cd results
poetry install --no-root
poetry run python3 plot.py
"
