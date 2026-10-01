#!/bin/bash
set -e

# usage: run_spadd_sparse.sh [threads] [dataset] [methods]
THREADS="${1:-16}"
DATASET="${2:-sparse}"
METHODS="${3:-serial_default_implementation graphblas_impl mkl_impl wingspan_spadd eigen_impl}"
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
# everything but birdseed; -t N,0: no interactive thread
julia --project=envs/wingspan -t $THREADS,0 spadd/run_spadd_wingspan.jl --dataset $DATASET --output spadd/results/spadd_${DATASET}_results.json --ncpu $THREADS $METHOD_FLAGS
"
