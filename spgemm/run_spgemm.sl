#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH --partition=lanka-v3
#SBATCH --qos=commit-main
#SBATCH --mem 102400

cd /data/scratch/paramuth/FinchBenchmarks/
bash instantiate_environments.sh

cd spgemm
bash run_spgemm.sh 12
