#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cpnest
import cpnest.model
import numpy as np
import h5py

# Gaussian noise around straight line with known sigma
sigma_x = 10
sigma_y = 50

class Inference(cpnest.model.Model):
    def __init__(self, pts):
        super(Inference,self).__init__()
        self.pts = pts
        self.names = ['m', 'c',]
        self.bounds = [[-50, 50], [-500, 500]]

    def log_prior(self, x):
        logP = super(Inference,self).log_prior(x)
        if np.isfinite(logP):
            return 0.0
        else:
            return -np.inf

    def log_likelihood(self, x):
        logL = np.sum(-0.5*((((self.pts[:,1]-x['c'])/x['m'])-self.pts[:,0])**2/(sigma_x**2+(sigma_y/x['m'])**2)))
        return logL

m = np.random.uniform(-5, 5)
c = np.random.uniform(-5, 5)
L = 100
npts = 10

import sys
if sys.argv[1] == '1':
    postprocess = True
else:
    postprocess = False

# Generate points
if not postprocess:
    x = np.random.uniform(-L/2, L/2, size = npts)
    y = m * x + c
    pts = np.column_stack([x + np.random.normal(0, sigma_x, size = npts), y + np.random.normal(0, sigma_y, size = npts)])
    np.savetxt('points.txt', pts)
    with open('true_values.txt', 'w') as f:
        f.write('m = '+str(m)+'\n')
        f.write('c = '+str(c)+'\n')
        f.write('L = '+str(L)+'\n')
else:
    pts = np.loadtxt('points.txt')
    with open('true_values.txt', 'r') as f:
        lines = f.readlines()
        m = float(lines[0].split('=')[1])
        c = float(lines[1].split('=')[1])
        L = float(lines[2].split('=')[1])

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
g.map_diag(sns.histplot)
g.map_lower(sns.histplot)

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
plt.errorbar(pts[:,0], pts[:,1], xerr=sigma_x, yerr = sigma_y, fmt = 'none', color = 'k')
x = np.linspace(-L/2-sigma_x*3, L/2+sigma_x*3, 10*L)
plt.plot(x, m*x + c, color = 'k', label = 'True', linestyle = '--')
f = np.array([samp[0]*x+samp[1] for samp in samps])
percs = np.percentile(f, [5, 16, 50, 84, 95], axis = 0)
plt.fill_between(x, percs[0], percs[-1], color = sns.color_palette()[0], alpha = 0.3)
plt.fill_between(x, percs[1], percs[-2], color = sns.color_palette()[0], alpha = 0.3)
plt.plot(x, percs[2], color = sns.color_palette()[0], label = 'Inference')

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()

plt.savefig('inference/fit_straight_line_function.pdf', dpi = 300)
