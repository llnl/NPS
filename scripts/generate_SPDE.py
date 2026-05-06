#!/bin/env python
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_size", help="\"Ns Nt Nx Ny [Nz for 3D]\"")
parser.add_argument("--method", help="laplacian or laplacian_sq or che or general_che or flux_che")
parser.add_argument("--T", default=1.0, type=float, help="temperature for general_che")
parser.add_argument('--saveT', action='store_true', help='add T as second channel')
parser.add_argument("--D", default=0.1, type=float, help="diffusivity")
parser.add_argument("--tskip", default=1, type=int, help="output every tskip steps")
parser.add_argument("--tskip_noise", default=1, type=int, help="adding noise every (default 1: every step)")
parser.add_argument("--tbegin", default=0, type=int, help="output from this step")
parser.add_argument("-o", default='output.npy', help="output.npy")
parser.add_argument("--c0", default='', help="if specified starting from this .npy file")
parser.add_argument("--noise", default=0, type=float, help="noise term")
parser.add_argument("--dc0", default=0.001, type=float, help="initial fluctuation")
parser.add_argument("--cmin", default=-0.5, type=float, help="cmin")
parser.add_argument("--cmax", default= 0.5, type=float, help="cmax")
parser.add_argument('--delta', action='store_true', help='output delta (useful when fitting delta with 2-frame seqences)')
parser.add_argument("--conservative_noise", default=1, type=int, help="make noise conservative")
parser.add_argument("--stiff", default=1, type=int, help="stiffness type")
parser.add_argument("--Trange", default="", type=str, help="e.g. \"0.1,0.9\". If T<0, sample random T in this range")
parser.add_argument("--entropy_order", default=15, type=int, help="Taylor expansion of entropy. -1 to use all terms with clipping. this option is NOT passed to the code yet.")
parser.add_argument("--lambda_bias", default=0.15, type=float, help="bias in double well potential")
parser.add_argument('--export_free_energy', action='store_true', help='output delta for free energy matching')

def laplacian(a):
    return np.roll(a,1,axis=0) + np.roll(a,-1,axis=0) + np.roll(a,1,axis=1) + np.roll(a,-1,axis=1) -4*a

def divergence_gradient(a, b):
    """
    div(a * grad(b))
    """
    return a*(np.roll(b,-1,0)-b) - np.roll(a,1,0)*(b-np.roll(b,1,0)) + a*(np.roll(b,-1,1)-b) - np.roll(a,1,1)*(b-np.roll(b,1,1))


def flux_eq_motion(mobility, mu, noise, dt, T):
    """
    equation of motion of concentration (conservative) driven by flux
    No. of flux directions = DIM (nearest neighbor. Assuming DIM=2)
    Output: dc (one implicit channel), fluxes (DIM channels)
    """
    # divergence_gradient(mobility(val[t-1], T), chem_pot(val[t-1], T) - stiffness(val[t-1], T)*laplacian(val[t-1]))
    # H_ij = np.stack([S_ij_func(c0, np.roll(c0,1,axis=i))*T for i in range(dim)], -1)
    # noise = np.random.randn(*(H_ij.shape)) * np.sqrt(H_ij)
    # H= np.sum([noise[...,i] - np.roll(noise[...,i],-1,axis=i) for i in range(dim)], axis=0)

    dmu = -mu[...,None] + nearest_nbs(mu)
    flux_mean = -mobility * dmu*(0.1/T)*dt
    flux_noise = np.random.randn(*flux_mean.shape)* np.sqrt(2*abs(mobility)*dt)*noise
    flux = flux_mean + flux_noise
    # print(np.mean(flux_mean), np.std(flux_mean), np.mean(flux_noise), np.std(flux_noise))
    return -flux.sum(-1) + np.roll(flux[...,0],1,0) + np.roll(flux[...,1],1,1), flux

def nearest_nbs(c):
    return np.stack([np.roll(c,-1, idir) for idir in range(2)],-1)

def mobility_flux(c, T):
    """
    mobility for each flux direction
    """
    # return (1+1.0*T)
    c1 = nearest_nbs(c)
    return ((((c+1)*(1-c))[...,None] + (c1+1)*(1-c1))/2+0.5)*(1+1.0*T)

def divergence_gradient_symm(a, b):
    """
    div(a * grad(b))
    """
    return ((np.roll(a,-1,0)+a)/2)*(np.roll(b,-1,0)-b) - ((a+np.roll(a,1,0))/2)*(b-np.roll(b,1,0)) + ((np.roll(a,-1,1)+a)/2)*(np.roll(b,-1,1)-b) - ((a+np.roll(a,1,1))/2)*(b-np.roll(b,1,1))

def PDE(modes, Nt, Nspace, method='laplacian', D=0.1, noise=0, cmin=-0.5, cmax=0.5, tskip_noise=1, c0=None, delta=False, dc0=0.001, T=1, saveT=False,
        conservative_noise=True, stiff=1, lambda_bias=0.15):
    dim=len(Nspace)
    xy= [np.arange(Ni)/Ni for Ni in Nspace]
    xy_grid = np.array(np.meshgrid(*xy)).transpose(np.roll(np.arange(len(xy)+1),-1))
    # val= np.zeros([Nt] + list(Nspace))
    val = [0] * Nt
    fluxes = []
    for t in range(Nt):
        if t == 0:
            if c0 is None:
                for x0, r0, magnitude in modes:
                    x=x0.reshape([1]*dim + [-1])
                    val[t]+= np.exp(-np.linalg.norm(xy_grid-x,axis=-1)**2/(2*r0**2))* magnitude
                if method in ['che', 'general_che', "flux_che", "allen_cahn"]:
                    val[t] = np.random.uniform(-dc0, dc0, Nspace) + np.random.uniform(cmin,cmax)
                    if T>0 and (method=='general_che'):
                        val[t] = np.clip(val[t], -0.996, 0.996)
            else:
                val[0] = c0
        else:
            if method=='laplacian':
                dx = laplacian(val[t-1])
            elif method=='laplacian_sq':
                dx = laplacian(laplacian(val[t-1]))
            elif method=='che':
                dx = laplacian(np.power(val[t-1],3)-val[t-1]-laplacian(val[t-1]))
            elif method=='general_che':
                dx = divergence_gradient(mobility(val[t-1], T), chem_pot(val[t-1], T) - stiffness(val[t-1], T, stiff)*laplacian(val[t-1]))
            elif method=='flux_che':
                dx, flux_i = flux_eq_motion(mobility_flux(val[t-1], T), chem_pot(val[t-1], T) - stiffness(val[t-1], T, stiff)*laplacian(val[t-1]), noise, D, T)
            elif method=='allen_cahn':
                dx = -(np.power(val[t-1],3)-val[t-1]-stiff*laplacian(val[t-1])) + lambda_bias
            else:
                raise 'ERROR unknown method '+method

            if method=="flux_che":
                val[t] = val[t-1] + dx
                fluxes.append(flux_i)
            else:
                val[t] = val[t-1] + D*dx if not delta else D*dx
                if noise != 0:
                    # val[t]+= diffusion_noise(val[t-1], dim)*noise
                    if t%tskip_noise==0:
                        if method=="allen_cahn":
                            val[t]+= np.sqrt(D)*noise*np.random.randn(*(val[t].shape))
                        else:
                            val[t]+= diffusion_noise(val[t-tskip_noise], dim, T, conservative_noise=conservative_noise)*noise

    val = np.stack(val)[...,None]
    if method == "flux_che":
        fluxes = [np.zeros_like(fluxes[0])] + fluxes
        fluxes = np.stack(fluxes)
        return np.concatenate([val, np.full_like(val, T), fluxes], -1)
    else:
        return np.concatenate([val, np.full_like(val, T)],-1) if saveT else val

def mobility(c, T): return (c+1)*(1-c)*(1+1.0*T)

def stiffness(c, T, stiff=1): return 1 + 2*T*(1-c**2/4) if stiff==1 else 0.5

entropy_coefficients = [[np.log(2), 0, -1/2, 0, -1/12, 0, -1/30, 0, -1/56, 0, -1/90, 0, -1/132, 0, -1/182],
                        [           0, -1/1 ,0,  -1/3, 0, -1/5 , 0, -1/7,  0,  -1/9, 0, -1/11,  0, -1/13]]
def setup_S_dS(order=-1):
    if order == -1:
        def _S_dS(c):
            c0 = 0.996
            c_safe = np.clip(c, -c0, c0)
            d_S = np.log((c_safe+1)/(1-c_safe))
            # S1 = ((1+c0)*np.log(1+c0)+(1-c0)*np.log(1-c0))
            dS1 = np.log((c0+1)/(1-c0))
            S = ((1+c_safe)*np.log(1+c_safe)+(1-c_safe)*np.log(1-c_safe) +
                (c>c0)*(c-c0)*dS1 - (c<-c0)*(c+c0)*dS1
                                )
            return np.log(2) - S/2, -d_S/2
        return _S_dS
    else:
        _S = Polynomial(entropy_coefficients[0][:order])
        _dS = Polynomial(entropy_coefficients[1][:order])
        def _S_dS(c):
            return _S(c), _dS(c)
        return _S_dS

S_dS = setup_S_dS(order=15)
# def chem_pot(c, T): return c**3-c + T*np.log((c+1)/(1-c))
def chem_pot(c, T):
    return c**3-c - T* S_dS(c)[1]

def free_energy(c, T):
    return c**4/4-c**2/2 - T* S_dS(c)[0]

# plt.plot(c, free_energy(c, 0.1), label="T=0.1");plt.plot(c, free_energy(c, 0.3), label="T=0.3");plt.plot(c, free_energy(c, 0.5), label="T=0.5");plt.legend(); plt.show()



# def S_ij_func(a,b): return np.maximum(np.abs(1-a**2), 0) * np.maximum(np.abs(1-b**2), 0)
def S_ij_func(a,b): return np.maximum(1-a**2, 0) * np.maximum(1-b**2, 0)

def diffusion_noise(c0, dim, T=1.0, conservative_noise=True):
    if not conservative_noise:
        raise NotImplementedError("noise")
    H_ij = np.stack([S_ij_func(c0, np.roll(c0,1,axis=i))*T for i in range(dim)], -1)
    noise = np.random.randn(*(H_ij.shape)) * np.sqrt(H_ij)
    H= np.sum([noise[...,i] - np.roll(noise[...,i],-1,axis=i) for i in range(dim)], axis=0)
    return H

def export_free_energy(Trange, crange, cell_shape, outf):
    # FE=np.concatenate([d,d[-2::-1]]).reshape(-1,1,1,1,1);
    cT = np.array(np.meshgrid(np.linspace(*crange,101), np.linspace(*Trange,31), indexing='ij')).reshape(2,-1).T.reshape((-1,1,1,2))
    cT=np.tile(cT,(1,3,3,1))
    # print(cT[...,0], cT[...,1])
    FE = free_energy(cT[...,0:1], cT[...,1:2])
    np.savez_compressed(outf.replace(".npy", ".npz"), np.concatenate((cT, FE),-1).astype(np.float32))
    #
    # np.save(outf, )


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
    if options.export_free_energy:
        export_free_energy(options.Trange, [options.cmin, options.cmax], Nspace, options.o)
        exit()
    for i in range(Ns):
        nmode = 4
        modes=[]
        if options.T < 0:
            T = np.random.uniform(*options.Trange)
        else:
            T = options.T
        for imode in range(nmode):
            x0= np.random.uniform(0.3, 0.7, dim)
            r0= np.random.uniform(0.04, 0.1)
            magnitude=np.random.uniform(1, 2)
            modes.append([x0, r0, magnitude])
        #print('debug i', i, modes)
        alldat.append(PDE(modes, Nt, Nspace, method=options.method, D=options.D, 
          dc0=options.dc0,
          lambda_bias=options.lambda_bias,
          cmin=options.cmin, cmax=options.cmax, noise=options.noise, tskip_noise=options.tskip_noise, c0=c0[i], delta=options.delta,
          conservative_noise=bool(options.conservative_noise), stiff=options.stiff,
          T=T, saveT=options.saveT))
    alldat = np.stack(alldat)
    np.save(options.o, alldat[:,options.tbegin::options.tskip].astype(np.float32))

# without noise: 
# $> parallel  python generate-SPDE.py --data_size '"5 10000 128 128"' --method che --D 0.01 --tskip 100 --tbegin 2800 -o ::: `seq 70`
# with noise: 
# $> parallel  python generate-SPDE.py --data_size '"10 10000 128 128"' --method che --D 0.01 --tskip 100 --tbegin 2800 --noise 0.01 --cmin -0.55 --cmax 0.55   -o ::: `seq 35`
# with noise tstep=1: 
# $> parallel  python generate-SPDE.py --data_size '"10 1000 64 64"' --method che --D 0.01 --tskip 1 --tbegin 2800 --noise 0.01 --cmin -0.55 --cmax 0.55   -o ::: `seq 35`
