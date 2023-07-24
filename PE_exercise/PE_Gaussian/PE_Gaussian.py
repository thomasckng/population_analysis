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

if __name__ == '__main__':
    mu = np.random.uniform(-10, 10)
    sigma = np.random.uniform(0, 5)
    npts = 100
    
    postprocess = False

    if not postprocess:
        # draw samples from a normal distribution
        samples = np.random.normal(mu, sigma, npts)
        np.savetxt('samples.txt', samples)
    else:
        samples = np.loadtxt('samples.txt')
    
    W = Inference(samples)
    if not postprocess:
        work = cpnest.CPNest(W, verbose = 2, output = 'inference/', nnest = 1, nensemble = 1, nlive = 1000)
        work.run()
        post = work.posterior_samples.ravel()
    else:
        with h5py.File('./inference/cpnest.h5', 'r') as f:
            post = f['combined']['posterior_samples'][()]
    samps   = np.column_stack([post[lab] for lab in ['mu', 'sigma']])
    
    # Plotting
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Plot samples histogram
    plt.hist(samples, bins = 20, density = True, color = sns.color_palette()[0], label = 'Data')

    # Plot Gaussian with true parameters
    x = np.linspace(-30, 30, 500)
    plt.plot(x, normal_distribution(x, mu, sigma), color = sns.color_palette()[3], label = 'True')

    # Plot Gaussian with maximum likelihood parameters
    max_likelihood = post[np.argmax(post['logL'])]
    plt.plot(x, normal_distribution(x, max_likelihood[0], max_likelihood[1]), color = sns.color_palette()[4], linestyle = '--', label = 'Maximum Likelihood')

    # Save plot
    plt.legend()
    plt.savefig('inference/PE_Gaussian.pdf')
