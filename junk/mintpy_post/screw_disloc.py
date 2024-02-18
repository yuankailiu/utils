#!/usr/bin/env python3
############################################################
# Screw dislocation model for interseismic deformation
# YKL @ 2021-05-20
############################################################

#%%
import numpy      as np
import matplotlib as mpl
from   scipy      import linalg
import matplotlib.pyplot   as plt

import pandas as pd
import matplotlib.pyplot as plt
import random
import datetime
import scipy
from scipy.optimize import minimize
from scipy.special import factorial as fac
from xamine_main import transec_pick

import emcee
import corner


def screw_disloc(params, x):
    v0, s, D = params
    v = v0 + (s/np.pi) * np.arctan(x/D)
    return v

def screw_RMS(params, x, data):
    s, D = params
    v0 = data - (s/np.pi) * np.arctan(x/D)
    v0 = 0# np.median(v0)
    params = [v0, s, D]
    pred = screw_disloc(params, x)
    rms = np.sqrt(np.mean((pred-data)**2))
    return rms

def log_prior(theta):
    s, D = theta
    s1 ,s2 = [-8, 0]
    D1 ,D2 = [0, 30]
    if s1 < s < s2 and D1 < D < D2:
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp - objFunc(theta)


pic_dir = './pic_supp'

# %%
file = './velocity_out/velocity_RlP.h5'
x,  z,  res  = transec_pick(file, 'velocity',    'prof_tmp.txt', fmt='lalo', mask_file='maskTempCoh.h5')
xe, ze, rese = transec_pick(file, 'velocityStd', 'prof_tmp.txt', fmt='lalo', mask_file='maskTempCoh.h5')

n = 5
center_shift = 50  # approx center of the fault, set as the distance origin
fig, axs = plt.subplots(figsize=[8, 36], nrows=int(len(res)/n), sharex=True,  gridspec_kw = {'wspace':0, 'hspace':0.05})
for i in range(int(len(res)/n)):
    ax = axs[i]
    ax.scatter(np.array(x)-center_shift, z, fc='whitesmoke', ec='lightgrey', alpha=0.2)
    xs = []
    zs = []
    zes= []
    for j in np.arange(i*n, (i+1)*n):
        xs = xs + list(res[j]['distance']/ 1000)
        zs = zs + list(res[j]['value']   * 1000)
        zes= zes+ list(rese[j]['value']  * 1000)
    xs  = np.array(xs)
    zs  = np.array(zs)
    zes = np.array(zes)
    markers, caps, bars = ax.errorbar(xs-center_shift, zs, yerr=2*zes, mfc='cornflowerblue', mec='k',
                                        fmt='o', errorevery=10, elinewidth=3, capsize=4, capthick=3)
    [bar.set_alpha(0.2) for bar in bars]
    [cap.set_alpha(0.2) for cap in caps]
    ax.set_xlim(0-center_shift, 120-center_shift)
    ax.set_ylim(-2.5, 2.5)
    ax.set_ylabel('LOS velo\n[mm/yr]')
    if ax == axs[-1]:
        ax.set_xlabel('Across-fault distance [km]')
plt.show()

#%% Let's guess the solution
i=7
xs = []
zs = []
zes= []
for j in np.arange(i*n, (i+1)*n):
    xs = xs + list(res[j]['distance']/ 1000)
    zs = zs + list(res[j]['value']   * 1000)
    zes= zes+ list(rese[j]['value']  * 1000)
idx = np.argsort(xs)
xs  = np.array(xs)[idx]
zs  = np.array(zs)[idx]
zs  = zs - np.mean(zs)
zes = np.array(zes)[idx]

#dist = np.linspace(0-center_shift,120-center_shift)
s0 =  -5    # mm/yr
D0 =   3    # km
v    = screw_disloc((0,s0*tolos,D0), xs-center_shift)
rms  = screw_RMS((s0*tolos,D0), xs-center_shift, zs)
plt.figure(figsize=[8,4])
plt.scatter(xs-center_shift, zs, ec='k', fc='cornflowerblue')
plt.plot(xs-center_shift, v, c='r', lw=2)
plt.text(0.001, 0.01, 's={}, D={}, rms={:.3f}'.format(s0, D0, rms), transform=ax.transAxes)
plt.ylim(-2.5, 2.5)
plt.xlim(0-center_shift, 120-center_shift)
plt.savefig(f'{pic_dir}/data_fit.png', dpi=150, bbox_inches='tight')

# %% Brutal grid search
tolos = np.cos(np.deg2rad(61))*np.sin(np.deg2rad(33))
s_bound = np.array([-8, 0])
D_bound = np.array([0, 30])

N = 40
M = 40
s_arr = np.linspace(s_bound[0], s_bound[1], N)
D_arr = np.linspace(D_bound[0], D_bound[1], M)
Rms = np.zeros([N,M])
for i in range(N):
    for j in range(M):
        Rms[i,j] = screw_RMS([s_arr[i]*tolos, D_arr[j]], xs-center_shift, zs)

plt.figure()
im = plt.imshow(Rms, extent=[D_bound[0],D_bound[-1],s_bound[-1],s_bound[0]], vmin=np.min(Rms), vmax=np.max(Rms), cmap='rainbow', aspect='equal')
cbar = plt.colorbar(im, shrink=0.4, label='RMS')
plt.savefig(f'{pic_dir}/param_space.png', dpi=150, bbox_inches='tight')


# %%

# Calc likelihood:
objFunc = lambda Params: screw_RMS(Params, xs-center_shift, zs)

disp = 0
method = 'SLSQP'

# MLE
my_fit = scipy.optimize.minimize(objFunc, np.array([s0, D0]), \
            bounds=np.array([s_bound, D_bound]), \
            tol = 1e-4, method=method, options={'disp': disp, 'maxiter':500})
print(my_fit)


pos0     = [-1, 5]
nwalkers = 32
ndim     = len(pos0)
pos      = pos0 + 1e-2 * np.random.randn(nwalkers, ndim)
sampler  = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, 5000, progress=True);

# %%
# Check out the chains
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ['s', 'D']
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], 'k', alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel('# Step')
plt.tight_layout()
plt.show()
# %%
