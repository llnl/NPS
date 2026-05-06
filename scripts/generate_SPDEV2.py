#!/bin/env python
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_SPDE import chem_pot

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_size", help="\"Nbatch,NTime,NX,NY[,NZ for 3D]\"")
parser.add_argument("--method", choices=["flux_che", "ace"])
parser.add_argument("--T", default=1.0, type=float, help="temperature for general_che")
parser.add_argument("--mob", default=0.1, type=float, help="mobility")
parser.add_argument("--Dt", default=0.1, type=float, help="time step")
parser.add_argument("--Dx", default=0.1, type=float, help="spatial grid size")
parser.add_argument("--stiff", default=0.5, type=float, help="lambda aka stiffness")
parser.add_argument("--tskip", default=1, type=int, help="Kt aka output every tskip steps")
parser.add_argument("--tbegin", default=0, type=int, help="output from this step")
parser.add_argument("-o", default='output.npy', help="output.npy")
parser.add_argument("--c0", default='', help="if specified starting from this .npy file")
parser.add_argument("--noise", default=0, type=float, help="epsilon aka noise term")
parser.add_argument("--dc0", default=0.001, type=float, help="initial fluctuation")
parser.add_argument("--cmin", default=-0.5, type=float, help="cmin")
parser.add_argument("--cmax", default= 0.5, type=float, help="cmax")
parser.add_argument("--Trange", default="", type=str, help="e.g. \"0.1,0.9\". If T<0, sample random T in this range")
parser.add_argument("--lambda_bias", default=0.15, type=float, help="ACE: bias in double well potential")

def laplacian(a, Dx):
    dim = a.ndim
    if dim == 2:
        return (np.roll(a,1,axis=0) + np.roll(a,-1,axis=0) + np.roll(a,1,axis=1) + np.roll(a,-1,axis=1) -2*dim*a)/Dx**dim
    elif dim == 3:
        return (np.roll(a,1,axis=0) + np.roll(a,-1,axis=0) + np.roll(a,1,axis=1) + np.roll(a,-1,axis=1) \
              + np.roll(a,1,axis=2) + np.roll(a,-1,axis=2) -2*dim*a)/Dx**dim

def flux_eq_motion(Mij, Bij, mu, Dt, Dx):
    """
    equation of motion of concentration (conservative) driven by flux
    No. of flux directions = DIM (nearest neighbor. Assuming DIM=2)
    Output: dc (one implicit channel), fluxes (DIM channels)
    """
    # print(f"debug {Mij=} {Bij=} {mu=} {Dt=} {Dx=}")
    dmu = (-mu[...,None] + nearest_nbs(mu))/Dx
    flux_mean = -Mij * dmu * Dt
    flux_noise = Bij * np.random.randn(*flux_mean.shape) * np.sqrt(Dt)
    flux = flux_mean + flux_noise
    # print(f"{abs(flux_mean).mean()=} {abs(flux_noise).mean()=}")
    # print(np.mean(flux_mean), np.std(flux_mean), np.mean(flux_noise), np.std(flux_noise))
    dim = mu.ndim
    if dim == 2:
        return -flux.sum(-1) + np.roll(flux[...,0],1,0) + np.roll(flux[...,1],1,1), flux
    elif dim == 3:
        return -flux.sum(-1) + np.roll(flux[...,0],1,0) + np.roll(flux[...,1],1,1) + np.roll(flux[...,2],1,2), flux

def nearest_nbs(c):
    dim = c.ndim
    return np.stack([np.roll(c,-1, idir) for idir in range(dim)],-1)

def mobility_flux(c, T, mob):
    """
    mobility for each flux direction
    """
    # return (1+1.0*T)
    c1 = nearest_nbs(c)
    return ((np.abs((c+1)*(1-c))[...,None] + np.abs((c1+1)*(1-c1)))/2+0.5)*(1+1.0*T) * mob


def PDE(Nt, Nspace, method='flux_che', mob=1.0, noise=0, cmin=-0.5, cmax=0.5, c0=None, dc0=0.001, T=1,
        Dt=0.1, Dx=0.1, return_flux=False,
         stiff=0.5, lambda_bias=0.15):
    dim=len(Nspace)
    xy= [np.arange(Ni)/Ni for Ni in Nspace]
    xy_grid = np.array(np.meshgrid(*xy)).transpose(np.roll(np.arange(len(xy)+1),-1))
    val = [0] * Nt
    fluxes = []
    for t in range(Nt):
        if t == 0:
            if c0 is None:
                val[t] = np.random.uniform(-dc0, dc0, Nspace) + np.random.uniform(cmin,cmax)
            else:
                val[0] = c0
        else:
            if method=='flux_che':
                Dij = mobility_flux(val[t-1], T, mob)
                dx, flux_i = flux_eq_motion(Dij/T, np.sqrt(2*Dij)*noise, chem_pot(val[t-1], T) - stiff*laplacian(val[t-1], Dx), Dt, Dx)
            elif method=='ace':
                dx = -mob*Dt*((np.power(val[t-1],3)-val[t-1]-stiff*laplacian(val[t-1], Dx)) - lambda_bias) + \
                    np.sqrt(Dt)*noise*np.random.randn(*(val[t].shape))
            else:
                raise 'ERROR unknown method '+method

            val[t] = val[t-1] + dx
            if method=="flux_che":
                fluxes.append(flux_i)

    val = np.stack(val)[...,None]
    if method == "flux_che" and return_flux:
        fluxes = [np.zeros_like(fluxes[0])] + fluxes
        fluxes = np.stack(fluxes)
        return np.concatenate([val, np.full_like(val, T), fluxes], -1)
    else:
        return np.concatenate([val, np.full_like(val, T)],-1)


if __name__ == "__main__":
    options = parser.parse_args()
    Ninput = list(map(int, options.data_size.replace(",", " ").split()))
    dim = len(Ninput) - 2
    Ns=Ninput[0]
    Nt=Ninput[1]
    Nspace=Ninput[2:]
    # alldat=np.zeros(Ninput + [2 if options.saveT else 1], dtype=np.float32)
    alldat = []
    c0=np.load(options.c0) if options.c0 else [None]*Ns
    if options.Trange:
        options.Trange = list(map(float, options.Trange.split(",")))[:2]
    for i in range(Ns):
        if options.T < 0:
            T = np.random.uniform(*options.Trange)
        else:
            T = options.T
        alldat.append(PDE(Nt, Nspace, method=options.method, mob=options.mob,
          dc0=options.dc0,
          lambda_bias=options.lambda_bias,
          cmin=options.cmin, cmax=options.cmax, noise=options.noise, c0=c0[i],
           stiff=options.stiff,
          T=T, Dx=options.Dx, Dt=options.Dt))
    alldat = np.stack(alldat)
    np.save(options.o, alldat[:,options.tbegin::options.tskip].astype(np.float32))
