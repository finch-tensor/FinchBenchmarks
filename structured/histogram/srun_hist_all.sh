#!/bin/bash
#SBATCH -J hist_all
#SBATCH -A gts-wahrens6
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH -t 8:00:00
#SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agushin3@gatech.edu
#SBATCH -C graniterapids
#SBATCH -o hist_all.out
#SBATCH -e hist_all.err

set -e

THREADS="${THREADS:-16}"

cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
numactl --cpunodebind=0 --membind=0 ./structured/histogram/run_hist_halide.sh "$THREADS"
numactl --cpunodebind=0 --membind=0 ./structured/histogram/run_hist_wingspan.sh "$THREADS"
numactl --cpunodebind=0 --membind=0 ./structured/histogram/run_hist_birdseed.sh "$THREADS"

# plot (no NUMA pinning needed)
./structured/histogram/plot_hist.sh "$THREADS"
