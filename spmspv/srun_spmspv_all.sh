#!/bin/bash
#SBATCH -J spmspv_all
#SBATCH -A gts-wahrens6
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH -t 3:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=psum3@gatech.edu
#SBATCH -C graniterapids
#SBATCH -o spmspv_all.out
#SBATCH -e spmspv_all.err

set -e

cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
./spmspv/run_spmspv.sh
./spmspv/run_spmspv_desc.sh
./spmspv/plot_spmspv.sh
