#!/bin/bash
#SBATCH -J spmspv_birdseed
#SBATCH -A gts-wahrens6
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH -t 3:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=psum3@gatech.edu
#SBATCH -C graniterapids
#SBATCH -o spmspv_birdseed.out
#SBATCH -e spmspv_birdseed.err

set -e

cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
./spmspv/run_spmspv_birdseed.sh

