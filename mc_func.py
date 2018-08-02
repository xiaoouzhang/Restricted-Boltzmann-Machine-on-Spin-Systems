import numpy as np

def d_energy(spin,x,y,new_ang,J):
    N=spin.shape[0]
    de=0
    up=np.mod(y+1,N)
    down=np.mod(y-1,N)
    left=np.mod(x-1,N)
    right=np.mod(x+1,N)
    de+=-J*(np.cos(new_ang-spin[left,y])-np.cos(spin[x,y]-spin[left,y]))
    de+=-J*(np.cos(new_ang-spin[right,y])-np.cos(spin[x,y]-spin[right,y]))
    de+=-J*(np.cos(new_ang-spin[x,up])-np.cos(spin[x,y]-spin[x,up]))
    de+=-J*(np.cos(new_ang-spin[x,down])-np.cos(spin[x,y]-spin[x,down]))
    return de

def d_energy_ising(spin,x,y,J):
    N=spin.shape[0]
    de=0
    up=np.mod(y+1,N)
    down=np.mod(y-1,N)
    left=np.mod(x-1,N)
    right=np.mod(x+1,N)
    de+=-J*(-spin[x,y]*spin[left,y]-spin[x,y]*spin[left,y])
    de+=-J*(-spin[x,y]*spin[right,y]-spin[x,y]*spin[right,y])
    de+=-J*(-spin[x,y]*spin[x,up]-spin[x,y]*spin[x,up])
    de+=-J*(-spin[x,y]*spin[x,down]-spin[x,y]*spin[x,down])
    return de

def update(spin,beta,J):
    #takes spin config and teperature
    #output new spin
    accept=0
    N=spin.shape[0]
    x=np.random.randint(N)
    y=np.random.randint(N)
    new_ang=np.random.rand()*np.pi*2.0
    de=d_energy(spin,x,y,new_ang,J)
    if de<0:
        spin[x,y]=new_ang
        accept=1
    else:
        rand=np.random.rand()
        if(rand<np.exp(-de*beta)):
            spin[x,y]=new_ang
            accept=1
            #print(new_ang-spin[x,y])
    return spin,accept

def update_ising(spin,beta,J):
    #takes spin config and teperature
    #output new spin
    accept=0
    N=spin.shape[0]
    x=np.random.randint(N)
    y=np.random.randint(N)
    
    de=d_energy_ising(spin,x,y,J)
    if de<0:
        spin[x,y]=-spin[x,y]
        accept=1
    else:
        rand=np.random.rand()
        if(rand<np.exp(-de*beta)):
            spin[x,y]=-spin[x,y]
            accept=1
    return spin,accept

def correlation(spin_all,N_corr):
    N_sample=spin_all.shape[0]
    N=spin_all.shape[1]
    corr=np.zeros(N_corr)
    ind=np.arange(0,N,int(N/N_corr))
    for i in np.arange(N_sample):
        for j in np.arange(N_corr):
            corr[j]+=np.cos(spin_all[i,0,ind[j]]-spin_all[i,0,0])
    corr=corr/N_sample
    return corr

def mc_chain(spin,N_sample,N,J,beta):
    np.random.seed()
    spin_sample=np.zeros((N_sample,N,N))
    for i in range(N_sample):
        for j in range(2000*N*N):
            [spin,_]=update(spin,beta,J)
        spin_sample[i,:,:]=spin
        #correlation[i,:]=mc.corre(spin,N_corr)
    return spin_sample

def mc_chain_ising(spin,N_sample,N,J,beta):
    np.random.seed()
    spin_sample=np.zeros((N_sample,N,N))
    for i in range(N_sample):
        for j in range(2000*N*N):
            [spin,_]=update_ising(spin,beta,J)
        spin_sample[i,:,:]=spin
    #correlation[i,:]=mc.corre(spin,N_corr)
    return spin_sample
