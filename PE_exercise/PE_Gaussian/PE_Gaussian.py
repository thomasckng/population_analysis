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
            return 0.0
        else:
            return -np.inf

    def log_likelihood(self, x):
        logL = np.sum(np.log(normal_distribution(self.samples, x['mu'], x['sigma'])))
        return logL


mu = np.random.uniform(-10, 10)
sigma = np.random.uniform(0, 5)
npts = 100

postprocess = True

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
    
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
sys.path.append('../../')
from kde_contour import kdeplot_2d_clevels, Bounded_1d_kde
sns.set_theme(palette='colorblind')

def kdeplot2d(x, y, rng=12345, **kws):
    kws.pop('label', None)
    kdeplot_2d_clevels(xs=x, ys=y, auto_bound=True, rng=rng, **kws)

def kdeplot1d(x, **kws):
    if np.all(x.isna()):
        return
    for key in ['label', 'hue_order', 'color']:
        kws.pop(key, None)
    df = pd.DataFrame({'x': x, 'y': Bounded_1d_kde(x, xlow=min(x), xhigh=max(x), **kws)(x)})
    df = df.sort_values(['x'])
    plt.fill_between(df['x'], df['y'], np.zeros(len(x)), alpha=0.2)
    plt.plot(df['x'], df['y'])
    plt.xlim(df['x'].min(), df['x'].max())
    current_ymax = plt.ylim()[1]
    if current_ymax > df['y'].max()*1.05:
        plt.ylim(0,current_ymax)
    else:
        plt.ylim(0,df['y'].max()*1.05)

df = pd.DataFrame(samps, columns = ['mu', 'sigma'])
vars = ['mu', 'sigma']
g = sns.PairGrid(df,
                 vars=vars,
                 corner=True,
                 diag_sharey=False,
                 layout_pad=0.
                )

g.map_lower(kdeplot2d, levels=[0.90,0.3935])
g.map_diag(kdeplot1d) 

for i in range(len(vars)):
    for j in range(i):
        g.axes[i,j].set_xlim(df[vars[j]].min(), df[vars[j]].max())
        g.axes[i,j].set_ylim(df[vars[i]].min(), df[vars[i]].max())

g.axes[1,0].set_xlabel(r"$\mu$")
g.axes[1,0].set_ylabel(r"$\sigma$")
g.axes[1,1].set_xlabel(r"$\mu$")

g.axes[0,0].axvline(mu, color = 'k', linestyle = '--')
g.axes[1,0].axvline(mu, color = 'k', linestyle = '--')
g.axes[1,0].axhline(sigma, color = 'k', linestyle = '--')
g.axes[1,1].axvline(sigma, color = 'k', linestyle = '--')

g.savefig('inference/PE_Gaussian.pdf', dpi=300)
