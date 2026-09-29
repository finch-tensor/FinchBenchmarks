#!/bin/bash
set -e

# usage: run_spadd_nacho.sh [threads] [datasets]
# threads only sets julia -t; the nacho kernel takes its thread count from TBB,
# which follows the CPUs Slurm gives the job.
THREADS="${1:-16}"
DATASETS="${2:-mirror sparse}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"

mkdir -p "$REPO_ROOT/.julia-depot" "$REPO_ROOT/.poetry-venv"

apptainer exec --cleanenv --no-home \
    --bind "$REPO_ROOT:/repo" \
    --env JULIA_DEPOT_PATH=/repo/.julia-depot \
    --env POETRY_VIRTUALENVS_PATH=/repo/.poetry-venv \
    --env JULIA_EXCLUSIVE=1 \
    "$SIF_IMAGE" bash -c "
cd /repo
make spadd/spadd_nacho
cd /repo/spadd
for d in $DATASETS; do
    julia -t $THREADS run_spadd.jl --dataset \$d --output results/spadd_nacho_\$d.json --ncpu $THREADS -m nacho_dcsr
done
"

# replot the speedup charts with nacho added next to the existing methods
"$SCRIPT_DIR/plot_spadd.sh" "$DATASETS"
