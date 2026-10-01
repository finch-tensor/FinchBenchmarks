#!/bin/bash
set -e

# usage: run_spadd_birdseed_mirror.sh [threads] [dataset]
MAX_THREADS="${1:-16}"
DATASET="${2:-mirror}"

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
    --env MAX_THREADS="$MAX_THREADS" \
    "$SIF_IMAGE" bash -c "
cd /repo
julia --project=envs/birdseed -t $MAX_THREADS,0 spadd/run_spadd_birdseed.jl -m birdseed_spadd -d $DATASET --ncpu $MAX_THREADS -o spadd/results/spadd_birdseed_$DATASET.json
"
