import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#return the average of h given x
def h_mean(x,w,b,sigma):
    #x: visible layer. One row is one data
    #w: weight, size_h*784
    #b: bias, size_h

    #return data*size_h
    b_shaped=np.reshape(b,(1,b.size))
    b_shaped=np.repeat(b_shaped,x.shape[0],axis=0)
    y=np.dot(x,w.T)/sigma+b_shaped
    return 1/(np.exp(-y)+1)


def x_mean(h,w,c,sigma):
    #h: visible layer. One row is one data
    #w: weight, size_h*784
    #c: bias, size_h

    #return data*size_h
    c_shaped=np.reshape(c,(1,c.size))
    c_shaped=np.repeat(c_shaped,h.shape[0],axis=0)
    y=np.dot(h,w)*sigma+c_shaped
    #return 1/(np.exp(-y)+1)
    return y
                
def gibbs_cd(x,kcd,w,b,c,sigma):
    h_p=h_mean(x,w,b,sigma)
    rad_h=np.random.rand(h_p.shape[0],h_p.shape[1])
    h_samp=np.zeros(h_p.shape)
    h_samp[rad_h<h_p]=1
    for i in range(kcd):
        #sample x
        x_p=x_mean(h_samp,w,c,sigma)
        x_samp=np.random.normal(x_p,sigma,x_p.shape)
        #sample h
        h_p=h_mean(x_samp,w,b,sigma)
        rad_h=np.random.rand(h_p.shape[0],h_p.shape[1])
        h_samp=np.zeros(h_p.shape)
        h_samp[rad_h<h_p]=1
    return x_samp

def h_mean_gaussian(x,w,b,sigma):
    b_shaped=np.reshape(b,(1,b.size))
    b_shaped=np.repeat(b_shaped,x.shape[0],axis=0)
    y=np.dot(x,w.T)+b_shaped
    return y

def x_mean_gaussian(h,w,c,sigma):
    c_shaped=np.reshape(c,(1,c.size))
    c_shaped=np.repeat(c_shaped,h.shape[0],axis=0)
    y=np.dot(h,w)+c_shaped
    #return 1/(np.exp(-y)+1)
    return y

def gibbs_cd_gaussian(x,kcd,w,b,c,sigma):
    h_p=h_mean_gaussian(x,w,b,sigma)
    h_samp=np.random.normal(h_p,sigma,h_p.shape)
    for i in range(kcd):
        #sample x
        x_p=x_mean_gaussian(h_samp,w,c,sigma)
        x_samp=np.random.normal(x_p,sigma,x_p.shape)
        #sample h
        h_p=h_mean_gaussian(x_samp,w,b,sigma)
        h_samp=np.random.normal(h_p,sigma,h_p.shape)
    return x_samp

def loss(x_in,x_p):
    l=(x_in-x_p)**2
    loss=np.sum(l)/x_in.size
    return loss

