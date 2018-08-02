import numpy as np
import random

#return the average of h given x
def h_mean(x,w,b):
    #x: visible layer. One row is one data
    #w: weight, size_h*784
    #b: bias, size_h

    #return data*size_h
    b_shaped=np.reshape(b,(1,b.size))
    b_shaped=np.repeat(b_shaped,x.shape[0],axis=0)
    y=np.dot(x,w.T)+b_shaped
    return 1.0/(np.exp(-y)+1.0)

def x_mean(h,w,c):
    #h: visible layer. One row is one data
    #w: weight, size_h*784
    #c: bias, size_h

    c_shaped=np.reshape(c,(1,c.size))
    c_shaped=np.repeat(c_shaped,h.shape[0],axis=0)
    y=np.dot(h,w)+c_shaped
    return np.tanh(y),1/(np.exp(-2.0*y)+1.0)

def gibbs_cd(x,kcd,w,b,c):
    h_p=h_mean(x,w,b)
    rad_h=np.random.rand(h_p.shape[0],h_p.shape[1])
    h_samp=np.zeros(h_p.shape)
    h_samp[rad_h<h_p]=1
    for i in range(kcd):
        #sample x
        _, x_prob=x_mean(h_samp,w,c)
        rad_x=np.random.rand(x_prob.shape[0],x_prob.shape[1])
        x_samp=-np.ones(x_prob.shape)
        x_samp[rad_x<x_prob]=1
        #sample h
        h_p=h_mean(x_samp,w,b)
        rad_h=np.random.rand(h_p.shape[0],h_p.shape[1])
        h_samp=np.zeros(h_p.shape)
        h_samp[rad_h<h_p]=1
    return x_samp

def loss(x_in,x_p):
    l=(x_in-x_p)**2
    loss=np.sum(l)/x_in.size
    return loss
