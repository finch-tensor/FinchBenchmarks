#!/bin/bash
set -e

MAX_THREADS="${1:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"

mkdir -p "$REPO_ROOT/.julia-depot" "$REPO_ROOT/.poetry-venv"
mkdir "$SCRIPT_DIR/results/weak"

apptainer exec --cleanenv --no-home \
    --bind "$REPO_ROOT:/repo" \
    --env JULIA_DEPOT_PATH=/repo/.julia-depot \
    --env POETRY_VIRTUALENVS_PATH=/repo/.poetry-venv \
    --env MAX_THREADS="$MAX_THREADS" \
    "$SIF_IMAGE" bash -c '
cd /repo/spgemm
THREADS=(1 2 4 8 16)
DATASETS=(w1 w2 w3 w4 w5)
for i in "${!THREADS[@]}"; do
    t="${THREADS[$i]}"
    [ "$t" -le "$MAX_THREADS" ] || break
    d="${DATASETS[$i]}"
    export OMP_NUM_THREADS="$t"
    export MKL_NUM_THREADS="$t"
    julia -t "$t" run_spgemm.jl --dataset "$d" --kernels fast --output "results/weak/spgemm_scale_threads_weak_${t}.json"
done
cd results
poetry install --no-root
poetry run python3 plot_weak_scaling.py
'
