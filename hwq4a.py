from mpi4py import MPI
import numpy as np
import time
import scipy.stats as sts
from scipy import optimize

def gs_rho(rho):
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  t0 = time.time()
  S=100
  T=1000
  mu = 3
  z_0 = mu
  np.random.seed(25)
  eps_mat = sts.norm.rvs(loc=0, scale=1, size=(S, T))

  N = int(S/size)
  z_mat = np.zeros([N, T])
  z_mat[0, :] = z_0
  neg_indx_sum = 0
  for s_ind in range(N):
    z_tm1 = z_0
    for t_ind in range(T):
        e_t = eps_mat[s_ind+N*rank, t_ind]
        z_t = rho * z_tm1 + (1 - rho) * mu + e_t
        #z_mat[s_ind, t_ind] = z_t
        z_tm1 = z_t
        if z_t<1:
          neg_indx_sum =  neg_indx_sum + t_ind
          break
  #print(neg_indx_sum)

  summ=comm.reduce(neg_indx_sum,op=MPI.SUM, root=0)
  if rank == 0:
    print("rho is " + str(rho)+" value is "+str(summ))
    return -summ
 
def main():
  t0 = time.time()
  print(optimize.minimize(gs_rho,x0 = 0.1, method = 'L-BFGS-B',
                          bounds = ((-0.95,0.95),),options = {'eps' : 0.05}))
  time_elapsed = time.time() - t0
  print("Time used: %d"% (time_elapsed))
if __name__ == '__main__':
    main()
