#!/bin/bash
set -e

# Strong scaling sweep: threads = 1, 2, 4, ... up to MAX_THREADS.
# usage: run_outer.sh [max_threads]
MAX_THREADS="${1:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"

mkdir -p "$REPO_ROOT/.julia-depot" "$REPO_ROOT/.poetry-venv" "$REPO_ROOT/outer_transpose/results"

apptainer exec --cleanenv --no-home \
    --bind "$REPO_ROOT:/repo" \
    --env JULIA_DEPOT_PATH=/repo/.julia-depot \
    --env POETRY_VIRTUALENVS_PATH=/repo/.poetry-venv \
    --env MAX_THREADS="$MAX_THREADS" \
    "$SIF_IMAGE" bash -c '
cd /repo
# generate the dataset once
[ -f outer_transpose/data/outer_A.ttx ] || (cd outer_transpose && julia --project=../envs/birdseed make_outer_data.jl)
t=1
while [ "$t" -le "$MAX_THREADS" ]; do
    export OMP_NUM_THREADS="$t"
    export MKL_NUM_THREADS="$t"
    # -t N,0: no interactive thread. mkl_0 (default path) is the accuracy reference.
    julia --project=envs/wingspan -t "$t",0 outer_transpose/run_outer_wingspan.jl --ncpu "$t" -o "outer_transpose/results/outer_wingspan_threads_${t}.json" -m mkl_0 -m wingspan_outer
    t=$((t * 2))
done
'
