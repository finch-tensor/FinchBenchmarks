#!/bin/bash
#SBATCH -J spadd_all
#SBATCH -A gts-wahrens6
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH -t 3:00:00
#SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agushin3@gatech.edu
#SBATCH -C graniterapids
#SBATCH -o spadd_all.out
#SBATCH -e spadd_all.err

set -e

cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
numactl --cpunodebind=0 --membind=0 ./spadd/run_spadd_mirror.sh
numactl --cpunodebind=0 --membind=0 ./spadd/run_spadd_sparse.sh
numactl --cpunodebind=0 --membind=0 ./spadd/run_spadd_birdseed_mirror.sh
numactl --cpunodebind=0 --membind=0 ./spadd/run_spadd_birdseed_sparse.sh
numactl --cpunodebind=0 --membind=0 ./spadd/run_spadd_nacho.sh
./spadd/plot_spadd.sh
