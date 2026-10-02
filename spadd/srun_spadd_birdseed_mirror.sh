#!/bin/bash
<<<<<<<< HEAD:spadd/srun_spadd_birdseed_mirror.sh
#SBATCH -J spadd_birdseed_mirror
========
#SBATCH -J spadd_birdseed
>>>>>>>> ps/nacho:spadd/srun_spadd_birdseed_sparse.sh
#SBATCH -A gts-wahrens6
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH -t 3:00:00
#SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agushin3@gatech.edu
#SBATCH -C graniterapids
<<<<<<<< HEAD:spadd/srun_spadd_birdseed_mirror.sh
#SBATCH -o spadd_birdseed_mirror.out
#SBATCH -e spadd_birdseed_mirror.err
========
#SBATCH -o spadd_birdseed.out
#SBATCH -e spadd_birdseed.err
>>>>>>>> ps/nacho:spadd/srun_spadd_birdseed_sparse.sh

set -e

cd "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
<<<<<<<< HEAD:spadd/srun_spadd_birdseed_mirror.sh
numactl --cpunodebind=0 --membind=0 ./spadd/run_spadd_birdseed_mirror.sh
========
./spadd/run_spadd_birdseed_sparse.sh
>>>>>>>> ps/nacho:spadd/srun_spadd_birdseed_sparse.sh

