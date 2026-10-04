#!/bin/bash
set -e

THREADS="${1:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"

mkdir -p "$REPO_ROOT/.julia-depot" "$REPO_ROOT/.poetry-venv"

apptainer exec --cleanenv --no-home \
    --bind "$REPO_ROOT:/repo" \
    --env JULIA_DEPOT_PATH=/repo/.julia-depot \
    --env POETRY_VIRTUALENVS_PATH=/repo/.poetry-venv \
    --env THREADS="$THREADS" \
    "$SIF_IMAGE" bash -c '
cd /repo
poetry install --no-root
make -C /repo/structured/histogram halide_kernel halide_rfactor_kernel NPROC_VAL="$THREADS"
cd /repo/structured/histogram
t="$THREADS"
export OMP_NUM_THREADS="$t"
export MKL_NUM_THREADS="$t"
# halide-atomics
julia --project=/repo/envs/wingspan -t "$t" run_hist.jl -a -m halide_hist --dataset image --ncpu "$t" --output "results/halide_hist_${t}_threads.json"
# halide-rfactor
julia --project=/repo/envs/wingspan -t "$t" run_hist.jl -a -m halide_rfactor_hist --dataset image --ncpu "$t" --output "results/halide_rfactor_hist_${t}_threads.json"
'
