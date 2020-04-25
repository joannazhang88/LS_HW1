# -*- coding: utf-8 -*-
"""HW1Q2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13aUhp5r7I9TYp6zdOGjhGyGN9QoclutU
"""



import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as clrand
import pyopencl.tools as cltools
from pyopencl.scan import GenericScanKernel
import matplotlib.pyplot as plt
import time

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
t0 = time.time()


n_runs = 1000
np.random.seed(25)
rho = 0.5 
sigma = 1.0 
z_0 = 3
S = 1000
T = int(4160)

rand_gen = clrand.PhiloxGenerator(ctx)
ran = rand_gen.normal(queue, (n_runs*T), np.float32, mu=3, sigma=1)

z_mat = np.zeros((T, N))
z_mat[0, :] = z_0


for s_ind in range(S): 
    z_tm1 = z_0
    for t_ind in range(T):
        e_t = ran[t_ind, s_ind]
        z_t = rho * z_tm1 + (1 - rho) * mu + e_t 
        z_mat[t_ind, s_ind] = z_t
        z_tm1 = z_t
average_finish = np.mean(z_mat[-1])
print(average_finish)

final_time = time.time()
time_elapsed = final_time - t0
print("Simulated %d Random Walks in: %f seconds"
                % (n_runs, time_elapsed))
