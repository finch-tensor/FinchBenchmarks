#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH --partition=lanka-v3
#SBATCH --qos=commit-main
#SBATCH --mem 102400
#SBATCH --array=0-8%9

cd /data/scratch/paramuth/FinchBenchmarks/spmv

bash run_spmv.sh 12
