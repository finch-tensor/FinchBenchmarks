#!/bin/bash
#SBATCH --account commit
#SBATCH --partition lanka-v3
#SBATCH --qos commit-main
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time 12:00:00
#SBATCH --mem 102400

cd /data/scratch/paramuth/FinchBenchmarks/
bash instantiate_environments.sh

cd spmv
bash run_spmv.sh 12
