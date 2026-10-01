#!/bin/bash
#SBATCH -J histogram
#SBATCH -A gts-wahrens6
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH -t 3:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=psum3@gatech.edu
#SBATCH -C graniterapids
#SBATCH -o histogram.out
#SBATCH -e histogram.err

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"

cd "$SLURM_SUBMIT_DIR"
./histogram/run_hist_wingspan.sh 16
./histogram/run_hist_birdseed.sh 16
./histogram/run_hist_halide.sh 16
./histogram/plot_hist.sh

