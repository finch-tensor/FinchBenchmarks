#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH --partition=lanka-v3
#SBATCH --qos=commit-main
#SBATCH --mem 102400

cd /data/scratch/paramuth/FinchBenchmarks/spadd

bash run_spadd.sh 12
