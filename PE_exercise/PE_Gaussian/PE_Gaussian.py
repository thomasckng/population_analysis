#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cpnest
import cpnest.model
import numpy as np
import h5py

def normal_distribution(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)

class Inference(cpnest.model.Model):
    def __init__(self, samples):
        super(Inference,self).__init__()
        self.samples = samples
        self.names = ['mu', 'sigma',]
        self.bounds = [[-10, 10], [0, 5]]

    def log_prior(self, x):
        logP = super(Inference,self).log_prior(x)
        if np.isfinite(logP):
            return np.log(1/x['sigma'])
        else:
            return -np.inf

    def log_likelihood(self, x):
        logL = np.sum(np.log(normal_distribution(self.samples, x['mu'], x['sigma'])))
        return logL


mu = np.random.uniform(-10, 10)
sigma = np.random.uniform(0, 5)
npts = 100

import sys
if sys.argv[1] == '1':
    postprocess = True
else:
    postprocess = False

# Generate points
if not postprocess:
    samples = np.random.normal(mu, sigma, npts)
    np.savetxt('samples.txt', samples)
    with open('true_values.txt', 'w') as f:
        f.write('mu = '+str(mu)+'\n')
        f.write('sigma = '+str(sigma)+'\n')
else:
    samples = np.loadtxt('samples.txt')
    with open('true_values.txt', 'r') as f:
        lines = f.readlines()
        mu = float(lines[0].split('=')[1])
        sigma = float(lines[1].split('=')[1])

# Inference
W = Inference(samples)
if not postprocess:
    work = cpnest.CPNest(W, verbose = 2, output = 'inference/', nnest = 1, nensemble = 1, nlive = 1000)
    work.run()
    post = work.posterior_samples.ravel()
else:
    with h5py.File('./inference/cpnest.h5', 'r') as f:
        post = f['combined']['posterior_samples'][()]
samps = np.column_stack([post[lab] for lab in ['mu', 'sigma']])
    
# Corner plot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
sys.path.append('../../')
sns.set_theme(palette='colorblind')

df = pd.DataFrame(samps, columns = ['mu', 'sigma'])
vars = ['mu', 'sigma']
g = sns.PairGrid(df,
                 vars=vars,
                 corner=True,
                 diag_sharey=False,
                 layout_pad=0.
                )

g.map_lower(sns.histplot)
g.map_diag(sns.histplot) 

g.axes[1,0].set_xlabel(r"$\mu$")
g.axes[1,0].set_ylabel(r"$\sigma$")
g.axes[1,1].set_xlabel(r"$\sigma$")

g.axes[0,0].axvline(mu, color = 'k', linestyle = '--')
g.axes[1,0].axvline(mu, color = 'k', linestyle = '--')
g.axes[1,0].axhline(sigma, color = 'k', linestyle = '--')
g.axes[1,1].axvline(sigma, color = 'k', linestyle = '--')

g.savefig('inference/PE_Gaussian_corner.pdf', dpi=300)

# Histogram
fig, ax = plt.subplots()
ax.hist(samples, bins=20, density=True, label='Data', color=sns.color_palette()[3])
x = np.linspace(samples.min()-5, samples.max()+5, 1000)
ax.plot(x, normal_distribution(x, mu, sigma), color='k', label='True', linestyle='--')
f = np.array([normal_distribution(x, samp[0], samp[1]) for samp in samps])
percs = np.percentile(f, [5, 16, 50, 84, 95], axis=0)
ax.fill_between(x, percs[0], percs[-1], color=sns.color_palette()[0], alpha=0.3)
ax.fill_between(x, percs[1], percs[-2], color=sns.color_palette()[0], alpha=0.3)
ax.plot(x, percs[2], color=sns.color_palette()[0], label='Inferred')

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$p(x)$')
ax.legend()

fig.savefig('inference/PE_Gaussian_function.pdf', dpi=300)
