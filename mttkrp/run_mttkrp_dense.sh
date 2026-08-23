#!/bin/bash
set -e

FULL="${1:-0}"
THREADS="${2:-16}"

if [ "$FULL" = "0" ]; then
    DATASET="large_short"
else
    DATASET="large"
fi

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
cd /repo/mttkrp
if [ ! -d data ] || [ -z \"\$(ls -A data 2>/dev/null)\" ]; then
    bash get_mttkrp_data.sh
fi
julia -t $THREADS run_mttkrp.jl --dataset $DATASET --output results/mttkrp_results.json --ncpu $THREADS
cd results
poetry install --no-root
poetry run python3 plot_mttkrp_results.py
"
