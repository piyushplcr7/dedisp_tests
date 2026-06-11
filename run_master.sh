#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 64
#SBATCH --time 0:10:0
#SBATCH --partition l40s
#SBATCH --gpus-per-task 8
#SBATCH --mem 330G
#SBATCH --exclusive
#SBATCH --job-name=run_master
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.out

#export OMP_PLACES=cores
#export OMP_PROC_BIND=closest

mkdir -p logs

module restore unified_dedisp

srun ../../../dedisp_master/build/bin/test/testdedisp -lodm 0 -dmstep 0.01 -numdms 1000 -multout -cleanout -nobary -o /scratch/panchal/masterout/out /scratch/panchal/chrisdata/*1.fits
