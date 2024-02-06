import numpy as np
from numba import njit

@njit
def truncated_powerlaw(m, alpha, mmin, mmax):
    p = m**-alpha * (alpha-1.)/(mmin**(1.-alpha)-mmax**(1.-alpha))
    p[m < mmin] = 0.
    p[m > mmax] = 0.
    return p

@njit
def smoothing(m, mmin, delta):
    p = np.zeros(m.shape, dtype = np.float64)
    p[m > mmin] = 1./(np.exp(delta/(m[m>mmin]-mmin) + delta/(m[m>mmin]-mmin-delta))+1)
    p[m >= mmin + delta] = 1.
    return p

@njit
def powerlaw_unnorm(m, alpha, mmin, mmax, delta):
    return truncated_powerlaw(m, alpha, mmin, mmax)*smoothing(m, mmin, delta)
    
@njit
def powerlaw(m, alpha, mmin, mmax, delta):
    x  = np.linspace(mmin, mmax, 1000)
    dx = x[1]-x[0]
    n  = np.sum(powerlaw_unnorm(x, alpha, mmin, mmax, delta)*dx)
    return powerlaw_unnorm(m.flatten(), alpha, mmin, mmax, delta)/n

@njit
def peak(m, mu, sigma):
    return np.exp(-0.5*(m-mu)**2/sigma**2)/(np.sqrt(2*np.pi)*sigma)

@njit
def plpeak(m, alpha, mmin, mmax, delta, mu, sigma, w):
    return (1.-w)*powerlaw(m, alpha, mmin, mmax, delta) + w*peak(m, mu, sigma)
