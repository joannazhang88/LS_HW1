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
from scipy import optimize

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

def get_neg_indx(rho, ran):
  rho = 0.5
  mu = 3
  S = 1000
  T = 4160
  eps_mat = sts.norm.rvs(loc=0, scale=1, size=(S*T)).astype(np.float32)
  initial = np.zeros(S).astype(np.float32)+3
  eps_mat = cl.array.to_device(queue, eps_mat)
  initial = cl.array.to_device(queue, initial)
  mknl = ElementwiseKernel(ctx,
       "float *a, float *b, float rho, float mu, float *rslt",
       "rslt[i] = rho * a[i] +(1-rho)*mu+b[i]" )    
  
  output = cl.array.empty_like(eps_mat)
  mknl(initial, eps_mat[:S], rho, mu, output[:S])
  for i in range(1,T):
    mknl(output[S*(i-1):S*i], eps_mat[S*i:S*i+1],rho, mu, output[S*i:S*(i+1)])
  z_mat = output.get().reshape(T,S)
  fst_neg_indx = np.zeros(S)
  for i in range(S):
     if np.all(z_mat.transpose()[i,:]>=0):
       pass
     else:
      fst_neg_indx[i] = np.where(z_mat.transpose()[i,:]<0)[0][0]
  return -np.sum(fst_neg_indx)
  
def main():
  t0 = time.time()
  S = 2
  T = 50
  ctx = cl.create_some_context()
  queue = cl.CommandQueue(ctx)
  rand_gen = clrand.PhiloxGenerator(ctx)
  ran = rand_gen.normal(queue, (S*T), np.float32, mu=0, sigma=1)
  print(optimize.minimize(get_neg_indx,x0 = 0.1,args = ran, method = 'L-BFGS-B',
                         bounds = ((-0.95,0.95),),options = {'eps' : 0.001}))
  time_elapsed = time.time() - t0
  print("Time used: %d"% (time_elapsed))

if __name__ == '__main__':
    main()
