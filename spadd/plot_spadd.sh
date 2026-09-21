#!/bin/bash
set -e

KERNELS="${1:-mirror sparse}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"

mkdir -p "$REPO_ROOT/.poetry-venv"

apptainer exec --cleanenv --no-home \
    --bind "$REPO_ROOT:/repo" \
    --env POETRY_VIRTUALENVS_PATH=/repo/.poetry-venv \
    "$SIF_IMAGE" bash -c "
cd /repo/spadd/results
poetry install --no-root
for k in $KERNELS; do
    poetry run python3 plot_spadd.py \$k
done
"
