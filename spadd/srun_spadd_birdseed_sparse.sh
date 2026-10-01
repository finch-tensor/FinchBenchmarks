#!/bin/bash
#SBATCH -J spadd_birdseed_sparse
#SBATCH -A gts-wahrens6
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH -t 3:00:00
#SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agushin3@gatech.edu
#SBATCH -C graniterapids
#SBATCH -o spadd_birdseed_sparse.out
#SBATCH -e spadd_birdseed_sparse.err

set -e

cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
numactl --cpunodebind=0 --membind=0 ./spadd/run_spadd_birdseed_sparse.sh

