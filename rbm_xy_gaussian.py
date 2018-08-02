import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import rbm_xy_fun as rbm_fun
import os

cwd=os.getcwd()

spin_all=np.load('sample_beta0_8.npy')

spin_flatten=np.zeros((spin_all.shape[0],spin_all.shape[1]**2))
for i in range(spin_all.shape[0]):
    spin_all[i,:,:]=np.mod(spin_all[i,:,:]-spin_all[i,:,:].mean(),2.0*np.pi)
    spin_flatten[i,:]=spin_all[i,:,:].flatten()

#learning rate
gamma=0.000000001
#momentum
beta=0.0
#l1 regularization
lambd=1.0

#number of hidden units
size_h=400
size_v=spin_flatten.shape[1]
w=np.reshape(np.random.normal(scale=0.1,size=size_v*size_h),(size_h,size_v))
#b=np.random.normal(scale=0.1,size=size_h)
#c=np.random.normal(scale=0.1,size=784)
b=np.zeros(size_h)
c=np.zeros(size_v)
#epoch number and cd number
epoch=200
batch=10
kcd=2
'''
w=np.load(cwd+'/parameters_rbm_gaussian/w_400_reg.npy')
b=np.load(cwd+'/parameters_rbm_gaussian/b_400_reg.npy')
c=np.load(cwd+'/parameters_rbm_gaussian/c_400_reg.npy')
'''
#std
sigma=0.5*2

train_loss=np.zeros(epoch)
grad_w0=np.zeros(w.shape)
grad_b0=np.zeros(b.shape)
grad_c0=np.zeros(c.shape)

np.seterr(all='raise')
print('start')

for i in range(epoch):
    spin_train=np.random.permutation(spin_flatten)
    for j in range(math.floor(spin_train.shape[0]/batch)):
        x_in=spin_train[j*batch:j*batch+batch,:]

        x_sampling=rbm_fun.gibbs_cd_gaussian(x_in,kcd,w,b,c,sigma)
        #gradient update
        grad_w=np.zeros(w.shape)
        grad_w=grad_w+(np.dot(rbm_fun.h_mean_gaussian(x_in,w,b,sigma).T,x_in)/sigma**2-np.dot(rbm_fun.h_mean_gaussian(x_sampling,w,b,sigma).T,x_sampling)/sigma**2)
        #print(grad_w)
        w=w+gamma*(grad_w+(beta*grad_w0)-lambd*np.sign(w))
        grad_w0=grad_w+beta*grad_w0
         
        grad_b=np.sum(rbm_fun.h_mean_gaussian(x_in,w,b,sigma)-rbm_fun.h_mean_gaussian(x_sampling,w,b,sigma),axis=0)/sigma**2
        b=b+gamma*(grad_b+beta*grad_b0)
        grad_b0=grad_b+beta*grad_b0
        
        grad_c=np.sum(x_in-x_sampling,axis=0)/sigma**2
        c=c+gamma*(grad_c+beta*grad_c0)
        grad_c0=grad_c+beta*grad_c0

    print('epoch'+str(i))

    #training reconstruction
    h_p=rbm_fun.h_mean_gaussian(spin_train,w,b,sigma)
    x_p=rbm_fun.x_mean_gaussian(h_p,w,c,sigma)
    train_loss[i]=rbm_fun.loss(spin_train,x_p)
    print('train'+str(train_loss[i]))


np.save(cwd+'/parameters_rbm_gaussian/w_400_reg.npy',w)
np.save(cwd+'/parameters_rbm_gaussian/b_400_reg.npy',b)
np.save(cwd+'/parameters_rbm_gaussian/c_400_reg.npy',c)
