#!/bin/bash

#SBATCH --job-name=q4
#SBATCH --output=q4a.out
#SBATCH --ntasks=50
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

module load mpi4py/3.0.1a0_py3


mpirun -n 20 python ./hw1q4a.py > ./q4a.out


wait
