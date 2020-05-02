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
S = 1000
T = 4160
rhos = np.linspace(-0.95, 0.95, num=200)
eps_mat = sts.norm.rvs(loc=0, scale=1, size=(S*T)).astype(np.float32)
initial = np.zeros(S).astype(np.float32)+3
eps_mat = cl.array.to_device(queue, eps_mat)
initial = cl.array.to_device(queue, initial)
mknl = ElementwiseKernel(ctx,
      "float *a, float *b, float rho, float mu, float *rslt",
      "rslt[i] = rho * a[i] +(1-rho)*mu+b[i]"
      )


average = np.empty(0)

for rho in rhos:
  output = cl.array.empty_like(eps_mat)
  mknl(initial, eps_mat[:S], rho, mu, output[:S])
  for k in range(1,T):
    mknl(output[S*(k-1):S*k], eps_mat[S*k:S*k+1],rho, mu, output[S*k:S*(k+1)])

  z_mat = output.get().reshape(T,S)
  fst_neg_indx = np.zeros(S)

  for i in range(S):
     if np.all(z_mat.transpose()[i,:]>=0):
       pass
     else:
      fst_neg_indx[i] = np.where(z_mat.transpose()[i,:]<0)[0][0]

  sum_result = np.sum(fst_neg_indx)
  average = np.append(average, (sum_result.get()/S).astype(float))
      
print(average)
time_elapsed = time.time() - t0
print(time_elapsed)
