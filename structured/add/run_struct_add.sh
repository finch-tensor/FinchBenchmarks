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
    --env MAX_THREADS="$MAX_THREADS" \
    "$SIF_IMAGE" bash -c '
cd /repo/structured/add
t=1
while [ "$t" -le "$MAX_THREADS" ]; do
    export OMP_NUM_THREADS="$t"
    export MKL_NUM_THREADS="$t"
    julia -t "$t" run_add.jl --dataset image --ncpu "$t" --output "results/add_${t}_threads.json"
    t=$((t * 2))
done
cd results
poetry install --no-root
poetry run python3 plot_add_strong_scaling.py
'
