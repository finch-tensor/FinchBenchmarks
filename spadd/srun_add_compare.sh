#!/bin/bash
#SBATCH -J struc
#SBATCH -A gts-wahrens6
#SBATCH -q embers
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH -t 3:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=psum3@gatech.edu
#SBATCH -C graniterapids
#SBATCH -o struc.out
#SBATCH -e struc.err

set -e

cd "$SLURM_SUBMIT_DIR"
./run_add_compare.sh

