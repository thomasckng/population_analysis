#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cpnest
import cpnest.model
import numpy as np
import matplotlib
import corner
import matplotlib.pyplot as plt
import corner
from figaro import plot_settings
from numba import njit
from scipy.stats import norm
import h5py

@njit
def log_normal(x, mu, sigma):
    return -0.5*((x-mu)/sigma)**2 - np.log(sigma) - 0.5*np.log(2*np.pi)

def log_pl(x, a, Mmin = 5, Mmax = 100):
    return np.log(a-1) - np.log( - Mmax**(-a+1) + Mmin**(-a+1)) -a*np.log(x)

@njit
def logL_jit(Mz, zs, z0, m0, sz, sm):
    return np.sum(log_normal(Mz/(1+zs), m0, sm) + log_normal(zs, z0, sz) - np.log(1+zs))

class Inference(cpnest.model.Model):

    def __init__(self, Mz):

        super(Inference,self).__init__()
        self.Mz = Mz
        self.names = ['z0', 'sz', 'm0', 'sm'] + ['z_{}'.format(i+1) for i in range(len(self.Mz))]
        self.bounds = [[0, 5], [0, 5], [5, 100], [0,10]] + [[0,10] for _ in range(len(self.Mz))]


    def log_prior(self, x):
        logP = super(Inference,self).log_prior(x)
        if np.isfinite(logP):
            return 0
        else:
            return -np.inf

    def log_likelihood(self, x):
        zs = np.array([x['z_{}'.format(i+1)] for i in range(len(self.Mz))])
        logL = logL_jit(self.Mz, zs, x['z0'], x['m0'], x['sz'], x['sm'])
        return logL

if __name__ == '__main__':
    
    z0 = 2
    sz  = 0.3
    m0 = 50
    sm = 5
    npts = 10
    
    postprocess = False
    
    if not postprocess:
        Mz = np.random.normal(m0, sm, size = npts)*(1+np.random.normal(z0, sz, size = npts))
        np.savetxt('Mz_samples.txt', Mz)
    else:
        Mz = np.loadtxt('Mz_samples.txt')
    
    
    W = Inference(Mz)
    if not postprocess:
        work = cpnest.CPNest(W, verbose = 2, output = 'inference/', nnest = 1, nensemble = 1, nlive = 1000)
        work.run()
        post = work.posterior_samples.ravel()
        samps = np.column_stack([post[lab] for lab in ['z0', 'sz', 'm0', 'sm']])
    else:
        with h5py.File('./inference/cpnest.h5', 'r') as f:
            samples = f['combined']['posterior_samples']
            samps   = np.column_stack([samples[lab] for lab in ['z0', 'sz', 'm0', 'sm']])
    # Plots
    fig = corner.corner(samps,
           labels=['$\\mu_z$', '$\\sigma_z$', '$\\mu_m$', '$\\sigma_m$'],
           quantiles=[0.05, 0.16, 0.5, 0.84, 0.95],
           truths = [z0, sz, m0, sm],
           show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
           use_math_text=True,
           filename='inference/joint_posterior.pdf')
    fig.savefig('inference/joint_posterior.pdf', bbox_inches='tight')
    
    m = np.linspace(0, 400, 1000)
    dm = m[1]-m[0]
    z = np.linspace(0, 7, 1000)
    probs = np.array([norm(samps[i,2], samps[i,3]).pdf(m/(1+z))*norm(samps[i,0], samps[i,1]).pdf(z) for i in range(len(samps))])
    
    percentiles = [50, 5, 16, 84, 95]
    p = {}
    for perc in percentiles:
        p[perc] = np.percentile(probs, perc, axis = 0)
    norm = p[50].sum()*dm
    for perc in percentiles:
        p[perc] = p[perc]/norm
    
    fig, ax = plt.subplots()
    ax.hist(Mz, bins = int(np.sqrt(len(Mz))), histtype = 'step', density = True)
    # CR
#    ax.fill_between(m, p[95], p[5], color = 'mediumturquoise', alpha = 0.25)
#    ax.fill_between(m, p[84], p[16], color = 'darkturquoise', alpha = 0.25)
    ax.plot(m, p[50], lw = 0.7, color = 'steelblue', label = '$\\mathrm{Median}$')
    ax.set_xlabel('$M_z\ [M_\\odot]$')
    ax.set_ylabel('$p(M_z)$')
    ax.legend(loc = 0)
    
    fig.savefig('inference/Mz_dist.pdf', bbox_inches = 'tight')
