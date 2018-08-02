#sampling from the RBM

import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import rbm_xy_fun as rbm_fun
import os

cwd=os.getcwd()
w=np.load(cwd+'/parameters_convrbm/w_6.npy')
b=np.load(cwd+'/parameters_convrbm/b_6.npy')
c=np.load(cwd+'/parameters_convrbm/c_6.npy')

spin_all=np.load('sample_beta0_8.npy')
spin_flatten=np.zeros((spin_all.shape[0],spin_all.shape[1]**2))

params={
        #number of hidden units becomes the # of filters
        'channel': 6,
        #filter size
        'k': 2,
        #lattice size
        'size_spin': spin_all.shape[1],
        #size of visible units
        'size_v': spin_flatten.shape[1],
        #std
        'sigma': 0.5*2,
        'kcd': 100
    }

x_sample_all=np.zeros((9,params['size_spin'],params['size_spin']))
for i in range(9):
    rad_x=np.random.rand(params['size_v'])
    x_sample=rbm_fun.conv_gibbs_cd(rad_x,w,b,c,params)
    x_sample_all[i,:,:]=np.reshape(x_sample,(params['size_spin'],params['size_spin']))

np.save('conv_rbm_spin.npy',x_sample_all)
