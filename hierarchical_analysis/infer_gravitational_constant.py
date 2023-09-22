#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cpnest
import cpnest.model
import numpy as np
import pandas as pd
import h5py

def gaussian(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

class Inference(cpnest.model.Model):
    def __init__(self, exp_data):
        super(Inference,self).__init__()
        self.exp_data = exp_data
        self.names = ['mu', 'sigma']
        self.bounds = [[6.66, 6.68], [0, 0.5]]

    def log_prior(self, x):
        logP = super(Inference,self).log_prior(x)
        if np.isfinite(logP):
            return np.log(1/x['sigma'])
        else:
            return -np.inf

    def log_likelihood(self, x):
        # log likelihood is the sum of the log likelihoods for each data point
        # the log likelihood for each data point is given by the log of the gaussian with mean x['mu'] and sigma sqrt(x['sigma']**2 + exp_data[i,1]**2) evaluated at exp_data[i,0]
        logL = np.sum(np.log(gaussian(self.exp_data[:,0], x['mu'], np.sqrt(x['sigma']**2 + self.exp_data[:,1]**2))))
        return logL

import sys
if sys.argv[1] == '1':
    postprocess = True
else:
    postprocess = False

# Load data
dat = pd.read_csv('newtonsConstantWithLabels.dat', sep = ' ', header = 0)
exp_data = dat[['#value', 'sigma']].to_numpy()

# Inference
W = Inference(exp_data)
if not postprocess:
    work = cpnest.CPNest(W, verbose = 2, output = 'inference/', nnest = 1, nensemble = 1, nlive = 1000)
    work.run()
    post = work.posterior_samples.ravel()
else:
    with h5py.File('./inference/cpnest.h5', 'r') as f:
        post = f['combined']['posterior_samples'][()]
samps = np.column_stack([post[lab] for lab in W.names])

# Corner plot
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme(palette='colorblind')

df = pd.DataFrame(samps, columns = W.names)
g = sns.PairGrid(df,
                 vars = W.names,
                 diag_sharey=False,
                 corner=True)
g.map_diag(sns.histplot)
g.map_lower(sns.histplot)

g.axes[1,0].set_xlabel('$\mu$')
g.axes[1,0].set_ylabel('$\sigma$')
g.axes[1,1].set_xlabel('$\sigma$')

g.savefig('inference/hierarchical_analysis_corner.pdf', dpi = 300)
plt.close()

# Function plot
x = np.linspace(6.668, 6.679, 1000)

f = np.array([gaussian(x, samp[0], samp[1]) for samp in samps])
percs = np.percentile(f, [5, 16, 50, 84, 95], axis = 0)
plt.fill_between(x, percs[0], percs[-1], color = sns.color_palette()[0], alpha = 0.3)
plt.fill_between(x, percs[1], percs[-2], color = sns.color_palette()[0], alpha = 0.3)
plt.plot(x, percs[2], color = sns.color_palette()[0], label = 'Inference')

# plot the original data as points with error bars at different heights
for i in range(len(exp_data)):
    plt.errorbar(exp_data[i,0], plt.ylim()[1]*0.7 - i*plt.ylim()[1]*0.7/len(exp_data), xerr = exp_data[i,1], fmt = 'x', color = 'k')

plt.ylim(0)
plt.xlabel(r'$G \, (10^{-11} \, \mathrm{m}^3 \, \mathrm{kg}^{-1} \, \mathrm{s}^{-2}$)')
plt.ylabel(r'$p(G)$')

plt.savefig('inference/hierarchical_analysis_posterior.pdf', dpi = 300)
