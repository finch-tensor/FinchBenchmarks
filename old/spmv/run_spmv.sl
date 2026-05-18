#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH --partition=lanka-v3
#SBATCH --qos=commit-main
#SBATCH --mem 102400
#SBATCH --array=0-8%9

cd /data/scratch/willow/FinchBenchmarks/spmv
source /afs/csail.mit.edu/u/w/willow/everyone/.bashrc

echo $SCRATCH
echo $JULIA_DEPOT_PATH
echo $JULIAUP_DEPOT_PATH
echo $PATH
echo $(pwd)
export TMPDIR=/tmp

datasets=(
    "willow_symmetric"
    "willow_unsymmetric"
    "permutation"
    "graph_symmetric"
    "graph_unsymmetric"
    "banded"
    "triangle"
    "taco_symmetric"
    "taco_unsymmetric"
)

dataset=${datasets[$SLURM_ARRAY_TASK_ID]}
julia run_spmv.jl -d $dataset -o split_results_${dataset}.json
