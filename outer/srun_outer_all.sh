#!/bin/bash
#SBATCH -J outer_all
#SBATCH -A gts-wahrens6
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH -t 8:00:00
#SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agushin3@gatech.edu
#SBATCH -C graniterapids
#SBATCH -o outer_all.out
#SBATCH -e outer_all.err

set -e

cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
numactl --cpunodebind=0 --membind=0 ./outer/run_outer.sh
numactl --cpunodebind=0 --membind=0 ./outer/run_outer_birdseed.sh

# plot (no NUMA pinning needed)
SIF_IMAGE="${SIF_IMAGE:-$PWD/wingspan_cgo27.sif}"
apptainer exec --cleanenv --no-home \
    --bind "$PWD:/repo" \
    --env POETRY_VIRTUALENVS_PATH=/repo/.poetry-venv \
    "$SIF_IMAGE" bash -c '
cd /repo/outer/results
poetry install --no-root
poetry run python3 plot_strong_scaling.py
'
