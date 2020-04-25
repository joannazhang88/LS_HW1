#!/bin/bash

#SBATCH --job-name=lifetime
#SBATCH --output=lifetime.out
#SBATCH --ntasks=1000
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

module load mpi4py/3.0.1a0_py3

for i in in `seq 1 1000`; do
	mpirun -i 1 python ./HW1Q1.py > ./mpi_hw$i.out
done

wait


	