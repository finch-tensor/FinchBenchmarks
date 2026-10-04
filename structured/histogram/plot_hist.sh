#!/bin/bash
set -e

THREADS="${1:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"

mkdir -p "$REPO_ROOT/.poetry-venv"

apptainer exec --cleanenv --no-home \
    --bind "$REPO_ROOT:/repo" \
    --env POETRY_VIRTUALENVS_PATH=/repo/.poetry-venv \
    --env THREADS="$THREADS" \
    "$SIF_IMAGE" bash -c '
cd /repo/structured/histogram/results
poetry install --no-root
poetry run python3 plot_hist_speedup.py "$THREADS"
'
