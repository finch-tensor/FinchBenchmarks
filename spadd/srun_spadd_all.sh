#!/bin/bash
#SBATCH -J spadd_all
#SBATCH -A gts-wahrens6
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH -t 3:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=psum3@gatech.edu
#SBATCH -C graniterapids
#SBATCH -o spadd_all.out
#SBATCH -e spadd_all.err

set -e

cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
./spadd/run_spadd_mirror.sh
./spadd/run_spadd_sparse.sh
./spadd/run_spadd_desc_mirror.sh
./spadd/run_spadd_desc_sparse.sh
./spadd/plot_spadd.sh
