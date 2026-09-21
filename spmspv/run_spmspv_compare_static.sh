#!/bin/bash
set -e

MAX_THREADS="${1:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"

mkdir -p "$REPO_ROOT/.julia-depot" "$REPO_ROOT/.poetry-venv"

apptainer exec --cleanenv --no-home \
    --bind "$REPO_ROOT:/repo" \
    --env JULIA_DEPOT_PATH=/repo/.julia-depot \
    --env POETRY_VIRTUALENVS_PATH=/repo/.poetry-venv \
    --env JULIA_EXCLUSIVE=1 \
    --env OMP_NUM_THREADS="$MAX_THREADS" \
    --env MKL_NUM_THREADS="$MAX_THREADS" \
    "$SIF_IMAGE" bash -c "
cd /repo
julia --project=. spmspv/run_spmspv_compare.jl -m coalesce_static -d snap_largest --threads $MAX_THREADS -o spmspv/results/spmspv_compare_static.json
"
