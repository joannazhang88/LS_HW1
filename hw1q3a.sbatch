#!/bin/bash

#SBATCH --job-name=q3
#SBATCH --output=q3a.out
#SBATCH --ntasks=100
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

module load mpi4py/3.0.1a0_py3


mpirun -n 20 python ./hw1q3a.py > ./q3a.out


wait
