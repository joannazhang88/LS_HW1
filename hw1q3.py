from mpi4py import MPI
import numpy as np
import time
import scipy.stats as sts

def gs_rho(rho):
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  t0 = time.time()
  S=1000
  T=4160
  mu = 3
  z_0 = mu
  np.random.seed(25)
  eps_mat = sts.norm.rvs(loc=0, scale=1, size=(S, T))

  N = int(S/size)
  z_mat = np.zeros([N, T])
  z_mat[0, :] = z_0

  negative = np.full(N,fill_value = T)
  for s_ind in range(N):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[s_ind+N*rank, t_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[s_ind, t_ind] = z_t
            z_tm1 = z_t
            if z_tm1<0:
              negative[s_ind] = t_ind
  
  #neg_indx_all = None
  #if rank == 0:
    #neg_indx_all = np.empty((N*size), dtype='float')
  #comm.Gather(sendbuf = negative, recvbuf = neg_indx_all, root=0)
  neg_indx_all = comm.gather(negative, root = 0)
  if rank == 0:
    return np.mean(neg_indx_all)

def main():
  rhos = np.linspace(-0.95, 0.95, num=20)
  sim_avg = np.zeros(20)
  #neg_results = np.zeros([20, S])
  for i in range(len(sim_avg)):
    sim_avg[i] = gs_rho(rhos[i])
  print(sim_avg)
if __name__ == '__main__':
    main()
