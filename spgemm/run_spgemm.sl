#!/bin/bash
#SBATCH --account commit
#SBATCH --partition lanka-v3
#SBATCH --qos commit-main
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time 12:00:00
#SBATCH --mem 0

cd /data/scratch/paramuth/FinchBenchmarks/
bash instantiate_environments.sh

cd spgemm
bash run_spgemm.sh 12
