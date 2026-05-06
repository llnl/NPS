#!/bin/env python
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_size", help="\"Ns Nt Nx Ny [Nz for 3D]\"")
parser.add_argument("--method", choices=("laplacian", "laplacian_sq", "che", "allen_cahn"))
parser.add_argument("--D", default=0.1, type=float, help="diffusivity")
parser.add_argument("--save_freq", default=1, type=int)
parser.add_argument("--nmode", default=4, type=int)
options = parser.parse_args()

def laplacian(a):
    return np.roll(a,1,axis=0) + np.roll(a,-1,axis=0) + np.roll(a,1,axis=1) + np.roll(a,-1,axis=1) -4*a


def PDE(modes, Nt, Nspace, method='laplacian', D=0.1):
    dim=len(Nspace)
    xy= [np.arange(Ni)/Ni for Ni in Nspace]
    xy_grid = np.array(np.meshgrid(*xy)).transpose(np.roll(np.arange(len(xy)+1),-1))
    val= np.zeros([Nt] + Nspace)
    for t in range(Nt):
        if t == 0:
            for x0, r0, magnitude in modes:
                x=x0.reshape([1]*dim + [-1])
                val[t]+= np.exp(-np.linalg.norm(xy_grid-x,axis=-1)**2/(2*r0**2))* magnitude
            if method=='che':
                val[t] = np.random.uniform(-0.001, 0.001,Nspace) + np.random.uniform(-0.3,0.3) # (val[t]-np.amin(val[t]))/(np.amax(val[t])-np.amin(val[t]))
            if method=='allen_cahn':
                val[t] = -1
                for x0, r0, magnitude in modes:
                    x=x0.reshape([1]*dim + [-1])
                    val[t]+= np.exp(-np.linalg.norm(xy_grid-x,axis=-1)**2/(2*r0**2)) * np.random.uniform(1.5,1.99)
        else:
            if method=='laplacian':
                dx = laplacian(val[t-1])
            elif method=='laplacian_0sq':
                dx = laplacian(laplacian(val[t-1]))
            elif method=='che':
                dx = laplacian(np.power(val[t-1],3)-val[t-1]-laplacian(val[t-1]))
            elif method=='allen_cahn':
                dx = -(np.power(val[t-1],3)-val[t-1]-laplacian(val[t-1])) + 0.15
            val[t]= val[t-1] + D*dx
    return val

Ninput = list(map(int, options.data_size.split()))
dim = len(Ninput) - 2
Ns=Ninput[0]
Ninput[1] *= options.save_freq
Nt=Ninput[1]
Nspace=Ninput[2:]
alldat=np.zeros(Ninput, dtype=np.float32)
for i in range(Ns):
    nmode = options.nmode
    modes=[]
    for imode in range(nmode):
        # x0= np.random.uniform(0.3, 0.7, dim)
        x0= np.random.uniform(0., 1, dim)
        r0= np.random.uniform(0.04, 0.1)
        magnitude=np.random.uniform(1, 2)
        modes.append([x0, r0, magnitude])
    #print('debug i', i, modes)
    alldat[i]= PDE(modes, Nt, Nspace, method=options.method, D=options.D)
np.save('output.npy', alldat[:,::options.save_freq, ..., None])
