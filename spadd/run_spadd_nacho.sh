#!/bin/bash
set -e

# usage: run_spadd_nacho.sh [threads] [datasets]
# The nacho kernel takes its thread count from TBB, which follows the process's
# CPU mask, so the run is pinned to the first $THREADS CPUs it may use (inside
# NUMA node 0 when launched under numactl) to match the Julia methods.
THREADS="${1:-16}"
DATASETS="${2:-sparse_largest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"

mkdir -p "$REPO_ROOT/.julia-depot" "$REPO_ROOT/.poetry-venv"

apptainer exec --cleanenv --no-home \
    --bind "$REPO_ROOT:/repo" \
    --env JULIA_DEPOT_PATH=/repo/.julia-depot \
    --env POETRY_VIRTUALENVS_PATH=/repo/.poetry-venv \
    "$SIF_IMAGE" bash -c "
cd /repo
make spadd/spadd_nacho
cd /repo/spadd
CPUS=\$(python3 -c 'import os; print(\",\".join(map(str, sorted(os.sched_getaffinity(0))[:$THREADS])))')
echo \"nacho pinned to CPUs \$CPUS\"
for d in $DATASETS; do
    taskset -c \$CPUS julia --project=../envs/birdseed -t $THREADS,0 run_spadd_birdseed.jl --dataset \$d --output results/spadd_nacho_\$d.json --ncpu $THREADS -m nacho_dcsr
done
"
