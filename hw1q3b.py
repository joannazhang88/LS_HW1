import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as clrand
import pyopencl.tools as cltools
from pyopencl.scan import GenericScanKernel
import matplotlib.pyplot as plt
import time
import scipy.stats as sts
from pyopencl.elementwise import ElementwiseKernel

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
t0 = time.time()

mu = 3
S = 2
T = 50
rhos = np.linspace(-0.95, 0.95, num=5)
eps_mat = sts.norm.rvs(loc=0, scale=1, size=(S*T)).astype(np.float32)
initial = np.zeros(S).astype(np.float32)+3
eps_mat = cl.array.to_device(queue, eps_mat)
initial = cl.array.to_device(queue, initial)
mknl = ElementwiseKernel(ctx,
      "float *a, float *b, float rho, float mu, float *rslt",
      "rslt[i] = rho * a[i] +(1-rho)*mu+b[i]"
      )
seg_boundaries = [1] + [0] * (S - 1) 
seg_boundaries = np.array(seg_boundaries, dtype=np.uint8)
seg_boundary_flags = cl_array.to_device(queue, seg_boundaries)
prefix_sum = GenericScanKernel(ctx, np.float32,
               arguments="__global float *ary, __global char *segflags,"
                   "__global float *out",
               input_expr="ary[i]",
               scan_expr="across_seg_boundary ? b : (a+b)", neutral="0",
               is_segment_start_expr="segflags[i]",
               output_statement="out[i] = item",
               options=[])
average = np.empty(0)
for rho in rhos:
  output = cl.array.empty_like(eps_mat)
  mknl(initial, eps_mat[:S], rho, mu, output[:S])
  for i in range(1,T):
    mknl(output[S*(i-1):S*i], eps_mat[S*i:S*i+1],rho, mu, output[S*i:S*(i+1)])
  z_mat = output.get().reshape(T,S)
  fst_neg_indx = np.empty(S)
  for i in range(S):
    fst_neg_indx[i] = np.where(z_mat.transpose()[i,:]<0)[0][0]
  sum_result = cl_array.empty(queue, shape = 1, dtype = float)
  prefix_sum(cl_array.to_device(queue, fst_neg_indx), seg_boundary_flags, sum_result)
  average = np.append(average, (sum_result.get()/S).astype(float))
print(average)
time_elapsed = time.time() - t0
print(time_elapsed)
