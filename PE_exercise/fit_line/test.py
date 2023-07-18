#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cpnest
import cpnest.model
import numpy as np

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

if __name__ == '__main__':
    
    m = np.random.uniform(-10, 10)
    c = np.random.uniform(-10, 10)
    npts = 10
    
    postprocess = False

    if not postprocess:
        x = np.random.uniform(-10, 10, size = npts)
        y = m*x + c + np.random.normal(0, sigma, size = npts)
        pts = np.column_stack([x, y])
        np.savetxt('points.txt', pts)
    else:
        pts = np.loadtxt('points.txt')
    
    W = Inference(pts)
    work = cpnest.CPNest(W, verbose = 2, output = 'inference/', nnest = 1, nensemble = 1, nlive = 1000)
    work.run()
    post = work.posterior_samples.ravel()
    samps = np.column_stack([post[lab] for lab in ['m', 'c']])

    print('m = {0:.3f}'.format(m))
    print('c = {0:.3f}'.format(c))
