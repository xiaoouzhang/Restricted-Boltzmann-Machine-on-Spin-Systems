#sampling from the rbm
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import rbm_xy_fun as rbm_fun
import os

data_d=os.getcwd()+'/parameters_rbm/'
w=np.genfromtxt(data_d+'w_400.txt',delimiter=',')
b=np.genfromtxt(data_d+'b_400.txt',delimiter=',')
c=np.genfromtxt(data_d+'c_400.txt',delimiter=',')
kcd=100
sigma=1

print('sigma=',sigma)
#show filters
fig, axs = plt.subplots(3,3, figsize=(7,7), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .001, wspace=.001)
axs = axs.ravel()
for i in range(9):
    rad_x=np.random.rand(1,c.size)
    x_sampling=rbm_fun.gibbs_cd(rad_x,kcd,w,b,c,sigma)
    h_p=rbm_fun.h_mean(x_sampling,w,b,sigma)
    x_p=rbm_fun.x_mean(h_p,w,c,sigma)
    x_shaped=np.reshape(x_p,(30,30))
    #axs[i].imshow(np.reshape(x_p,(30,30)),cmap='Greys')
    axs[i].quiver(np.cos(x_shaped),np.sin(x_shaped),pivot='mid')
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
plt.savefig('rbm_sample.pdf')
np.save('rbm_spin.npy',x_shaped)        
