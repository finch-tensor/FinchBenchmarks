#!/bin/bash
set -e

# usage: run_spmspv_birdseed.sh [threads] [methods]
MAX_THREADS="${1:-16}"
METHODS="${2:-birdseed-bytemap-static birdseed-bytemap-dynamic birdseed-hash-static birdseed-hash-dynamic}"
METHOD_FLAGS=""
for m in $METHODS; do METHOD_FLAGS="$METHOD_FLAGS -m $m"; done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"

mkdir -p "$REPO_ROOT/.julia-depot" "$REPO_ROOT/.poetry-venv"

# -t N,0: no interactive thread (Julia 1.12's -t N adds one, oversubscribing the cores)
apptainer exec --cleanenv --no-home \
    --bind "$REPO_ROOT:/repo" \
    --env JULIA_DEPOT_PATH=/repo/.julia-depot \
    --env POETRY_VIRTUALENVS_PATH=/repo/.poetry-venv \
    --env JULIA_EXCLUSIVE=1 \
    --env OMP_NUM_THREADS="$MAX_THREADS" \
    --env MKL_NUM_THREADS="$MAX_THREADS" \
    "$SIF_IMAGE" bash -c "
cd /repo
julia --project=envs/birdseed -t $MAX_THREADS,0 spmspv/run_spmspv_birdseed.jl -d snap_largest -o spmspv/results/spmspv_birdseed.json $METHOD_FLAGS
"
