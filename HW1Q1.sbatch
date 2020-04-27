#!/bin/bash

#SBATCH --job-name=lifetime
#SBATCH --output=lifetime.out
#SBATCH --ntasks=120
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

module load mpi4py/3.0.1a0_py3


#mpirun -n 1000 python ./HW1Q1.py > ./lifetime.out
for i in {1...100}
do 
   mpirun -n $i python ./HW1Q1.py
done

wait


	
