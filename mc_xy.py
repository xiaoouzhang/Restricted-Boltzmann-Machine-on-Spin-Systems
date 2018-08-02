import numpy as np
from joblib import Parallel, delayed
import mc_func as mc

#lattice size
N=30
spin=np.random.rand(N,N)*np.pi*2.0
N_sample=5
#spin_sample=np.zeros((N_sample,N,N))
N_corr=10
correlation=np.zeros((N_sample,N_corr))
J=1.0
#beta=1.2*1000
beta=0.9

#go to equilibrium
for i in range(3000*N*N):
    [spin,_]=mc.update(spin,beta/10,J)
for i in range(3000*N*N):
    [spin,_]=mc.update(spin,beta/5,J)
for i in range(3000*N*N):
    [spin,_]=mc.update(spin,beta/2,J)
for i in range(3000*N*N):
    [spin,_]=mc.update(spin,beta,J)

print('start')
#sampling

result = Parallel(n_jobs=6)(delayed(mc.mc_chain)(spin,N_sample,N,J,beta) for chain in range(6))

print('sampling finished')
spin_all=result[0]
del result[0]
for s in result:
    spin_all=np.concatenate((spin_all,s),axis=0)

#calculation correlation function
corr=mc.correlation(spin_all,N_corr)



np.save('corr.npy',corr)
#np.save('sample.npy',spin_sample)
np.save('sample_all_beta09.npy',spin_all)
