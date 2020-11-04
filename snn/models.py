import numpy as np
import matplotlib.pyplot as plt

def delayed_rectifier(v,dt,nchannels=1):

    """Simulates a delayed recififer channel"""

    alpha_n = 0.01*(v+55)/(1-np.exp(-0.1*(v+55))) #opening rate
    beta_n = 0.125*np.exp(-0.0125*(v+65)) #closing rate

    n = np.zeros((nchannels, *v.shape))

    for channel in range(nchannels):
        for i in range(len(v)-1):
            n[channel,i+1] = \
            n[channel,i]+(alpha_n[i]*(1-n[channel,i])-beta_n[i]*n[channel,i])*dt

    p = n**4
    i = np.random.binomial(1, p)

    return p,i
