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
cd /repo/structured/histogram
t="$THREADS"
export OMP_NUM_THREADS="$t"
export MKL_NUM_THREADS="$t"
julia --project=/repo/envs/birdseed -t "$t" run_hist_birdseed.jl -m birdseed_hist --dataset image --ncpu "$t" --output "test.json" -a
'
