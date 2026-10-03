#!/bin/bash
set -e

# usage: run_outer_birdseed.sh [threads]
THREADS="${1:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"

mkdir -p "$REPO_ROOT/.julia-depot" "$REPO_ROOT/.poetry-venv" "$REPO_ROOT/outer/results"

apptainer exec --cleanenv --no-home \
    --bind "$REPO_ROOT:/repo" \
    --env JULIA_DEPOT_PATH=/repo/.julia-depot \
    --env POETRY_VIRTUALENVS_PATH=/repo/.poetry-venv \
    --env JULIA_EXCLUSIVE=1 \
    --env OMP_NUM_THREADS="$THREADS" \
    --env MKL_NUM_THREADS="$THREADS" \
    "$SIF_IMAGE" bash -c "
cd /repo
# generate the dataset once
[ -f outer/data/outer_A.ttx ] || (cd outer && julia --project=../envs/birdseed make_outer_data.jl)
# -t N,0: no interactive thread
julia --project=envs/birdseed -t $THREADS,0 outer/run_outer_birdseed.jl --ncpu $THREADS -o outer/results/outer_birdseed.json -m birdseed_outer
"
