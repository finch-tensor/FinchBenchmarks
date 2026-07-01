#!/bin/bash
#SBATCH -J struc
#SBATCH -A gts-wahrens6
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH -t 3:00:00
#SBATCH -C graniterapids
#SBATCH -o struc.out
#SBATCH -e struc.err

export OMP_NUM_THREADS=16

cd add
./run.sh 16

cd ..

cd histogram
./run.sh 16