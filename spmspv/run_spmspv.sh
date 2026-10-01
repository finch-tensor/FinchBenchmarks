#!/bin/bash
set -e

# usage: run_spmspv.sh [threads] [methods]
THREADS="${1:-16}"
METHODS="${2:-graphblas eigen wingspan-bytemap-static wingspan-bytemap-dynamic wingspan-hash-static wingspan-hash-dynamic}"
METHOD_FLAGS=""
for m in $METHODS; do METHOD_FLAGS="$METHOD_FLAGS -m $m"; done

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"

mkdir -p "$REPO_ROOT/.julia-depot" "$REPO_ROOT/.poetry-venv"

apptainer exec --cleanenv --no-home \
    --bind "$REPO_ROOT:/repo" \
    --env JULIA_DEPOT_PATH=/repo/.julia-depot \
    --env POETRY_VIRTUALENVS_PATH=/repo/.poetry-venv \
    --env JULIA_EXCLUSIVE=1 \
    --env OMP_NUM_THREADS="$THREADS" \
    --env MKL_NUM_THREADS="$THREADS" \
    "$SIF_IMAGE" bash -c "
cd /repo
julia --project=envs/wingspan -t $THREADS,0 spmspv/run_spmspv_wingspan.jl --dataset snap_largest --output spmspv/results/spmspv_wingspan.json $METHOD_FLAGS
"
