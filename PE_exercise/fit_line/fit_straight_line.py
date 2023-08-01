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
        logL = np.sum(np.log(normal_distribution(self.pts[:,0], (self.pts[:,1] - x['c']) / x['m'], sigma) * normal_distribution(self.pts[:,1], x['m'] * self.pts[:,0] + x['c'], sigma)))
        return logL

m = np.random.uniform(-5, 5)
c = np.random.uniform(-5, 5)
npts = 10

import sys
if sys.argv[1] == '1':
    postprocess = True
else:
    postprocess = False

# Generate points
if not postprocess:
    x = np.random.uniform(-10, 10, size = npts)
    y = m * x + c
    pts = np.column_stack([x + np.random.normal(0, sigma, size = npts), y + np.random.normal(0, sigma, size = npts)])
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

# Corner plot
import seaborn as sns
import matplotlib.pyplot as plt
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

g.savefig('inference/fit_straight_line_corner.pdf', dpi = 300)
plt.close()

# Function plot
plt.scatter(pts[:,0], pts[:,1], color = 'k', label = 'Data')
plt.errorbar(pts[:,0], pts[:,1], xerr=sigma, yerr = sigma, fmt = 'none', color = 'k')
x = np.linspace(-15, 15, 100)
plt.plot(x, m*x + c, color = 'k', label = 'True', linestyle = '--')
plt.plot(x, np.median(samps[:,0])*x + np.median(samps[:,1]), color = 'r', label = 'Inferred')
plt.fill_between(x, np.percentile(samps[:,0], 16)*x + np.percentile(samps[:,1], 16), np.percentile(samps[:,0], 84)*x + np.percentile(samps[:,1], 84), color = 'r', alpha = 0.2)

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()

plt.savefig('inference/fit_straight_line_function.pdf', dpi = 300)
