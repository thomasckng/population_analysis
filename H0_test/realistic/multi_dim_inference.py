import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon as jsd
from scipy.optimize import minimize
from figaro.mixture import DPGMM
from figaro.utils import rejection_sampler, get_priors
from figaro.cosmology import CosmologicalParameters
from figaro.load import load_density
from multiprocessing import Pool

def PLpeak(m, alpha = -2., mmin = 5., mmax = 70., mu = 30., sigma = 4., w = 0.2):
    norm_pl = (1-alpha)/(mmin**(alpha+1) - mmax**(alpha+1))
    pl      = norm_pl * m**alpha
    peak    = np.exp(-0.5*((m-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
    return w*pl + (1-w)*peak

# DL distributions
def DLsq(DL, DLmax = 5000):
    return 3*DL**2/DLmax**3

def realistic_jsd(x, i):
    z = CosmologicalParameters(x[0]/100., 0.315, 0.685, -1., 0., 0.).Redshift(dL)
    m = np.einsum("i, j -> ij", mz, np.reciprocal(1+z))
    
    p_realistic = np.einsum("ij, j -> ij", PLpeak(m, mu=x[1]), DLsq(dL))
    p_realistic = np.trapz(p_realistic, dL, axis=1)
    p_realistic = p_realistic/np.trapz(p_realistic, mz, axis=0)

    return jsd(p_realistic, realistic_figaro[i])

def scipy_minimize(i):
    return minimize(realistic_jsd, x0=[np.random.uniform(20,60), np.random.uniform(10,50)], bounds = ((20,60),(10,50)), args=(i,)).x

if __name__ == '__main__':
    true_H0 = 40. # km/s/Mpc (Deliberately off value)

    n_draws_samples = 1000
    n_draws_figaro = 1000
    M_min = 0
    M_max = 200

    # Generate samples from source distribution
    valid = False
    while not valid:
        dL_sample = rejection_sampler(n_draws_samples, DLsq, [0, 5000])
        M_sample  = rejection_sampler(n_draws_samples, PLpeak, [5,70])
        z_sample  = np.array([CosmologicalParameters(true_H0/100., 0.315, 0.685, -1., 0., 0.).Redshift(d) for d in dL_sample])
        Mz_sample = M_sample * (1 + z_sample)
        valid = Mz_sample.max() < M_max and Mz_sample.min() > M_min

    mz = np.linspace(10,np.max(Mz_sample) * 1.05,1000)
    dL = np.linspace(10,5000,500)

    mix = DPGMM([[M_min, M_max]], prior_pars=get_priors([[M_min, M_max]], Mz_sample))
    realistic_figaro = np.array([mix.density_from_samples(Mz_sample) for _ in tqdm(range(n_draws_figaro))])
    realistic_figaro = np.array([realistic_figaro[i].pdf(mz) for i in range(len(realistic_figaro))])

    with Pool(64) as p:
        result = p.map(scipy_minimize, range(len(realistic_figaro)))

    np.savez("multi_dim_result.npz", result = result, Mz_sample = Mz_sample, realistic_figaro = realistic_figaro)
