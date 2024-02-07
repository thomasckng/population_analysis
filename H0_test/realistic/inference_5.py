import numpy as np
from numpy.random import uniform
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon as jsd
from scipy.optimize import minimize
from figaro.mixture import DPGMM
from figaro.utils import rejection_sampler, get_priors
from figaro.cosmology import CosmologicalParameters
from multiprocessing import Pool
from numba import njit

# Mass distribution
# @njit
def truncated_powerlaw(m, alpha, mmin, mmax):
    p = m**-alpha * (alpha-1.)/(mmin**(1.-alpha)-mmax**(1.-alpha))
    p[m < mmin] = 0.
    p[m > mmax] = 0.
    return p

# @njit
def smoothing(m, mmin, delta):
    p = np.zeros(m.shape, dtype = np.float64)
    p[m > mmin] = 1./(np.exp(delta/(m[m>mmin]-mmin) + delta/(m[m>mmin]-mmin-delta))+1)
    p[m >= mmin + delta] = 1.
    return p

# @njit
def powerlaw_unnorm(m, alpha, mmin, mmax, delta):
    return truncated_powerlaw(m, alpha, mmin, mmax)*smoothing(m, mmin, delta)
    
# @njit
def powerlaw(m, alpha, mmin, mmax, delta):
    x  = np.linspace(mmin, mmax, 1000)
    dx = x[1]-x[0]
    n  = np.sum(powerlaw_unnorm(x, alpha, mmin, mmax, delta)*dx)
    return powerlaw_unnorm(m, alpha, mmin, mmax, delta)/n

# @njit
def peak(m, mu, sigma):
    return np.exp(-0.5*(m-mu)**2/sigma**2)/(np.sqrt(2*np.pi)*sigma)

# @njit
def plpeak(m, alpha=3.5, mmin=5, mmax=90, delta=5, mu=35, sigma=5, w=0.2):
    return (1.-w)*powerlaw(m, alpha, mmin, mmax, delta) + w*peak(m, mu, sigma)

# dL distributions
# @njit
def DLsq(DL, DLmax = 5000):
    return 3*DL**2/DLmax**3

def reconstruct_observed_distribution(samples):
    mix = DPGMM([[M_min, M_max]], prior_pars=get_priors([[M_min, M_max]], samples))
    return mix.density_from_samples(samples)

def realistic_jsd(x, i):
    z = CosmologicalParameters(x[0]/100., 0.315, 0.685, -1., 0., 0.).Redshift(dL)
    m = np.einsum("i, j -> ij", mz, np.reciprocal(1+z))
    
    p_realistic = np.einsum("ij, j -> ij", plpeak(m, alpha=x[1], mu=x[2], sigma=x[3], w=x[4]), DLsq(dL))
    p_realistic = np.trapz(p_realistic, dL, axis=1)
    p_realistic = p_realistic/np.trapz(p_realistic, mz, axis=0)

    return jsd(p_realistic, realistic_figaro[i])

def scipy_minimize(i):
    return minimize(realistic_jsd,
                    x0=[uniform(10,100), uniform(1,5), uniform(10,50), uniform(0,10), uniform(0,1)],
                    bounds = ((10,100), (1,5), (10,50), (0,10), (0,1)), args=(i,)).x

if __name__ == '__main__':
    true_H0 = 40. # km/s/Mpc (Deliberately off value)

    n_draws_samples = 1000
    n_draws_figaro = 10000
    M_min = 0
    M_max = 200

    # Generate samples from source distribution
    valid = False
    while not valid:
        dL_sample = rejection_sampler(n_draws_samples, DLsq, [0,5000])
        M_sample  = rejection_sampler(n_draws_samples, plpeak, [20,200])
        z_sample  = np.array([CosmologicalParameters(true_H0/100., 0.315, 0.685, -1., 0., 0.).Redshift(d) for d in dL_sample])
        Mz_sample = M_sample * (1 + z_sample)
        valid = Mz_sample.max() < M_max and Mz_sample.min() > 20 # prevent overflow with Mz > 20

    mz = np.linspace(np.min(Mz_sample),np.max(Mz_sample),1000)
    dL = np.linspace(10,5000,500)

    with Pool(64) as p:
        realistic_figaro = p.map(reconstruct_observed_distribution, np.full((n_draws_figaro,len(Mz_sample)), Mz_sample))

    realistic_figaro = np.array([realistic_figaro[i].pdf(mz) for i in range(len(realistic_figaro))])

    with Pool(64) as p:
        result = p.map(scipy_minimize, range(len(realistic_figaro)))

    np.savez("result_5_param.npz", result = result, Mz_sample = Mz_sample, realistic_figaro = realistic_figaro)
