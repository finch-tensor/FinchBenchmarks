#!/bin/bash
set -e

THREADS="${1:-16}"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"

apptainer exec --cleanenv --no-home --writable-tmpfs \
    --bind "$REPO_ROOT:/repo" \
    --env POETRY_VIRTUALENVS_PATH=/opt/poetry-venv \
    --env JULIA_DEPOT_PATH=/opt/julia-depot \
    --env OMP_NUM_THREADS="$THREADS" \
    --env MKL_NUM_THREADS="$THREADS" \
    "$SIF_IMAGE" bash -c "
cd /repo/hadamard
julia -t $THREADS run_hadamard.jl --dataset mirror --output results/hadamard_mirror_results.json --ncpu $THREADS
cd results
poetry run python3 plot.py mirror
"
