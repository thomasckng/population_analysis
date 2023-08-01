#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cpnest
import cpnest.model
import numpy as np
import h5py

# Gaussian noise around straight line with known sigma
sigma = 1

def normal_distribution(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)

class Inference(cpnest.model.Model):
    def __init__(self, pts):
        super(Inference,self).__init__()
        self.pts = pts
        self.names = ['m', 'c',]
        self.bounds = [[-10, 10], [-10, 10]]

    def log_prior(self, x):
        logP = super(Inference,self).log_prior(x)
        if np.isfinite(logP):
            return 0.0
        else:
            return -np.inf

    def log_likelihood(self, x):
        logL = np.sum(np.log(normal_distribution(self.pts[:,1], x['m']*self.pts[:,0] + x['c'], sigma)))
        return logL

m = np.random.uniform(-10, 10)
c = np.random.uniform(-10, 10)
npts = 10

postprocess = True

# Generate points
if not postprocess:
    x = np.random.uniform(-10, 10, size = npts)
    y = m*x + c + np.random.normal(0, sigma, size = npts)
    pts = np.column_stack([x, y])
    np.savetxt('points.txt', pts)
    with open('true_values.txt', 'w') as f:
        f.write('m = '+str(m)+'\n')
        f.write('c = '+str(c)+'\n')
else:
    pts = np.loadtxt('points.txt')
    with open('true_values.txt', 'r') as f:
        lines = f.readlines()
        m = float(lines[0].split('=')[1])
        c = float(lines[1].split('=')[1])

# Inference
W = Inference(pts)
if not postprocess:
    work = cpnest.CPNest(W, verbose = 2, output = 'inference/', nnest = 1, nensemble = 1, nlive = 1000)
    work.run()
    post = work.posterior_samples.ravel()
else:
    with h5py.File('./inference/cpnest.h5', 'r') as f:
        post = f['combined']['posterior_samples'][()]
samps = np.column_stack([post[lab] for lab in ['m', 'c']])

# Plotting
import seaborn as sns
import pandas as pd
sns.set_theme(palette='colorblind')

df = pd.DataFrame(samps, columns = ['m', 'c'])
g = sns.PairGrid(df,
                 vars = ['m', 'c'],
                 diag_sharey=False,
                 corner=True)
g.map_diag(sns.kdeplot, fill=True)
g.map_lower(sns.kdeplot, levels=[1-0.9, 1-0.3935])

g.axes[1,0].set_xlabel(r'$m$')
g.axes[1,0].set_ylabel(r'$c$')
g.axes[1,1].set_xlabel(r'$c$')

g.axes[0,0].axvline(m, color = 'k', linestyle = '--')
g.axes[1,0].axvline(m, color = 'k', linestyle = '--')
g.axes[1,0].axhline(c, color = 'k', linestyle = '--')
g.axes[1,1].axvline(c, color = 'k', linestyle = '--')

g.savefig('inference/fit_straight_line.pdf', dpi = 300)
