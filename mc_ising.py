import numpy as np
from joblib import Parallel, delayed
import mc_func as mc

#lattice size
N=30
init_rand=np.random.rand(N,N)-0.5
spin=np.sign(init_rand)
N_sample=5

N_corr=10
correlation=np.zeros((N_sample,N_corr))
J=1.0

beta=0.35

#go to equilibrium

for i in range(3000*N*N):
    [spin,_]=mc.update_ising(spin,beta,J)


print('start')
#sampling

result = Parallel(n_jobs=6)(delayed(mc.mc_chain_ising)(spin,N_sample,N,J,beta) for chain in range(6))

print('sampling finished')
spin_all=result[0]
del result[0]
for s in result:
    spin_all=np.concatenate((spin_all,s),axis=0)

#calculation correlation function
corr=mc.correlation(spin_all,N_corr)


np.save('ising_beta035.npy',spin_all)
