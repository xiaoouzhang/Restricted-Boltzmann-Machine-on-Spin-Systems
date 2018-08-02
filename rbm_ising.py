import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import rbm_ising_fun as rbm_fun
import os

cwd=os.getcwd()
#read data
spin_all=np.load('ising_beta035.npy')
spin_flatten=np.zeros((spin_all.shape[0],spin_all.shape[1]**2))
for i in range(spin_all.shape[0]):
    spin_flatten[i,:]=spin_all[i,:,:].flatten()

#learning rate
gamma=0.0005
#momentum
beta=0.0
#l1 regularization
lambd=0.1

#number of hidden units
size_h=400
size_v=spin_flatten.shape[1]
w=np.reshape(np.random.normal(scale=0.1,size=size_v*size_h),(size_h,size_v))
#b=np.random.normal(scale=0.1,size=size_h)
#c=np.random.normal(scale=0.1,size=784)
b=np.zeros(size_h)
c=np.zeros(size_v)
#epoch number and cd number
epoch=100
batch=10
kcd=2


train_loss=np.zeros(epoch)
grad_w0=np.zeros(w.shape)
grad_b0=np.zeros(b.shape)
grad_c0=np.zeros(c.shape)

print('start')

for i in range(epoch):
    spin_train=np.random.permutation(spin_flatten)
    for j in range(math.floor(spin_train.shape[0]/batch)):
        x_in=spin_train[j*batch:j*batch+batch,:]

        x_sampling=rbm_fun.gibbs_cd(x_in,kcd,w,b,c)
        #gradient update
        grad_w=np.zeros(w.shape)
        for k in range(batch):
            x_k=np.reshape(x_in[k,:],(1,x_in[k,:].size))
            x_ksample=np.reshape(x_sampling[k,:],(1,x_sampling[k,:].size))
            grad_w=grad_w+(np.outer(rbm_fun.h_mean(x_k,w,b),x_k)-np.outer(rbm_fun.h_mean(x_ksample,w,b),x_ksample))
        
        
        w=w+gamma*(grad_w-lambd*np.sign(w))
        b=b+gamma*np.sum(rbm_fun.h_mean(x_in,w,b)-rbm_fun.h_mean(x_sampling,w,b),axis=0)
        
        
        c=c+gamma*np.sum(x_in-x_sampling,axis=0)

    print('epoch'+str(i))

    #training reconstruction
    h_p=rbm_fun.h_mean(spin_train,w,b)
    x_p,_=rbm_fun.x_mean(h_p,w,c)
    train_loss[i]=rbm_fun.loss(spin_train,x_p)
    print('train'+str(train_loss[i]))


np.save(cwd+'/parameters_rbm/w_ising.npy',w)
np.save(cwd+'/parameters_rbm/b_ising.npy',b)
np.save(cwd+'/parameters_rbm/c_ising.npy',c)
