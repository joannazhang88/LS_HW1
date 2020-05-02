# -*- coding: utf-8 -*-

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
np.random.seed(25)
rho = 0.5 
sigma = 1.0 
z_0 = 3
S = 1000
T = int(4160)

rand_gen = clrand.PhiloxGenerator(ctx)
ran = rand_gen.normal(queue, (S*T), np.float32, mu=0, sigma=1)
      
seg_boundaries = [1] + [0] * (T - 1) 
seg_boundaries = np.array(seg_boundaries, dtype=np.uint8)
seg_boundary_flags = np.tile(seg_boundaries, int(S))
seg_boundary_flags = cl_array.to_device(queue, seg_boundary_flags)

prefix_sum = GenericScanKernel(ctx, np.float32,
               arguments="__global float *ary, __global char *segflags, "
                   "__global float *out",
               input_expr="segflags[i] == 1 ? ary[i]: ary[i]",
               scan_expr="across_seg_boundary ? b : (a*0.5+b+1.5)", neutral="0",
               is_segment_start_expr="segflags[i]",
               output_statement="out[i] = item",
               options=[])

dev_result = cl_array.empty_like(ran)
prefix_sum(ran, seg_boundary_flags, dev_result)
r_walks_all = (dev_result.get()
                         .reshape(S, T)
                         .transpose()
                         )

average_finish = np.mean(z_mat[-1])
#print(average_finish)
final_time = time.time()
time_elapsed = final_time - t0
print("1000 simultaions in: %f seconds"
                % (time_elapsed))

#Another approach with Elementwise Kernels
  # eps_mat = sts.norm.rvs(loc=0, scale=1, size=(S*T)).astype(np.float32)
  # initial = np.zeros(S).astype(np.float32)+3
  # eps_mat = cl.array.to_device(queue, eps_mat)
  # initial = cl.array.to_device(queue, initial)
  # mknl = ElementwiseKernel(ctx,
  #     "float *a, float *b, float rho, float mu, float *rslt",
  #     "rslt[i] = 0.5 * a[i] +1.5+b[i]"
  #     )
