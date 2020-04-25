#!/bin/bash

#SBATCH --job-name=lifetime
#SBATCH --output=lifetime.out
#SBATCH --ntasks=1000
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

module load mpi4py/3.0.1a0_py3


mpirun -n 1000 python ./HW1Q1.py > ./lifetime.out


wait


	
