#!/bin/bash
#SBATCH -J spadd_largest
#SBATCH -A gts-wahrens6
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH -t 8:00:00
#SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agushin3@gatech.edu
#SBATCH -C graniterapids
#SBATCH -o spadd_largest.out
#SBATCH -e spadd_largest.err

# mirror_largest / sparse_largest (spmspv's snap_largest matrices):
# birdseed, wingspan, nacho, and mkl (the plot's baseline).
set -e

THREADS=16
DATASETS="sparse_largest"

cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
for d in $DATASETS; do
    kernel="${d%_largest}"   # mirror or sparse
    numactl --cpunodebind=0 --membind=0 ./spadd/run_spadd_$kernel.sh $THREADS $d "mkl_impl wingspan_spadd"
    numactl --cpunodebind=0 --membind=0 ./spadd/run_spadd_birdseed_$kernel.sh $THREADS $d
done
numactl --cpunodebind=0 --membind=0 ./spadd/run_spadd_nacho.sh $THREADS "$DATASETS"
./spadd/plot_spadd.sh "$DATASETS"
