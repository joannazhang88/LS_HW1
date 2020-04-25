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

z_mat = np.zeros((T, S), dtype=np.float32)
z_mat[0, :] = z_0

rand_gen = clrand.PhiloxGenerator(ctx)
ran = rand_gen.normal(queue, (T*n_runs), np.float32, mu=3, sigma=1)

for s_ind in range(2): 
    z_tm1 = z_0
    for t_ind in range(2):
        e_t = ran[s_ind*T+t_ind]
        z_t = rho * z_tm1 + (1 - rho) * 3 + e_t
        z_mat[t_ind, s_ind] = z_t
        z_tm1 = z_t
        
#seg_boundaries = [1] + [0]*(T-1)
#seg_boundaries = np.array(seg_boundaries, dtype=np.uint8)
#seg_boundary_flags = np.tile(seg_boundaries, int(n_runs))
#seg_boundary_flags = cl_array.to_device(queue, seg_boundary_flags)

#prefix_sum = GenericScanKernel(ctx, np.float32,
#                arguments="__global float *ary, __global char *segflags, "
#                    "__global float *out",
#                input_expr="ary[i]0.5+(0.5*3)",
#                scan_expr="across_seg_boundary ? b : (a+b)", neutral="0",
#                is_segment_start_expr="segflags[i]",
#                output_statement="out[i] = item + 100",
#                options=[])





average_finish = np.mean(z_mat[-1])
print(average_finish)

final_time = time.time()
time_elapsed = final_time - t0
print("Simulated %d Random Walks in: %f seconds"
                % (n_runs, time_elapsed))
