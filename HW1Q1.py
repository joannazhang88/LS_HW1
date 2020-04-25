#!/usr/bin/env python
# coding: utf-8

# In[2]:


from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as sts


# In[ ]:


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# In[ ]:


N = int(1000/size)
t0 = time.time()

rho = 0.5 
mu = 3.0 
sigma = 1.0 
z_0 = mu
S = 1000
T = int(4160)
np.random.seed(25)

eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, N)) 
z_mat = np.zeros((T, N))
z_mat[0, :] = z_0


for s_ind in range(N): 
    z_tm1 = z_0
    for t_ind in range(T):
        e_t = eps_mat[t_ind, s_ind]
        z_t = rho * z_tm1 + (1 - rho) * mu + e_t 
        z_mat[t_ind, s_ind] = z_t
        z_tm1 = z_t


# In[6]:


#simulations = None
if rank == 0:
    comm.Gather(z_mat, root=0)


# In[ ]:


time_elapsed = time.time() - t0
#print(time_elapsed)
print("Simulated lifetimem in: %f seconds on %d MPI processes"
                % (time_elapsed, size))   

