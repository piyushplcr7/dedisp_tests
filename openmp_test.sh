#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 64
#SBATCH --gpus-per-task 8
#SBATCH --time 1:0:0
#SBATCH --partition l40s
#SBATCH --mem 100G
#SBATCH --job-name=openmp_fdd
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.out

#module restore dedisp_omp_jed

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-$(nproc)}

ARGS="-lodm 0 -dmstep 0.01 -numdms 1000 -multout -nobary"
mkdir -p /scratch/panchal/cpuout
srun ./build/bin/test/testdedisp $ARGS \
	-o /scratch/panchal/cpuout/outnew \
	/scratch/panchal/chrisdata/*1.fits

