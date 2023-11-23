import numpy as np

"""
README

The double_gaussian is
p(M, DL) = p(M|DL)p(DL) = evolving_gaussian(M, DL)*gaussian(DL)

The realistic model is
p(M, DL) = p(M)p(DL) = PLpeak(M)*DLsq(DL)
"""

# Mass distributions
def PLpeak(m, alpha = -2., mmin = 5., mmax = 70., mu = 30., sigma = 4., w = 0.2):
    norm_pl = (1-alpha)/(mmin**(alpha+1) - mmax**(alpha+1))
    pl      = norm_pl * m**alpha
    peak    = np.exp(-0.5*((m-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
    return w*pl + (1-w)*peak

def evolving_gaussian(m, DL):
    mu  = mean(DL)
    std = sigma(DL)
    return np.exp(-0.5*((m-mu)/std)**2)/(np.sqrt(2*np.pi)*std)

def mean(DL, dMdDL = 30./1000., offset = 5.):
    return DL*dMdDL + offset

def std(DL, dMdDL = 8./1000., offset = 1.):
    return DL*dMdDL + offset

# DL distributions
def DLsq(DL, DLmax = 5000):
    return 3*DL**2/DLmax**3

def gaussian(DL, mu = 1600, sigma = 400):
    return np.exp(-0.5*((DL-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
