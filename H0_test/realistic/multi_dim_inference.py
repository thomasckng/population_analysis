import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon as jsd
from scipy.optimize import minimize
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

mz = np.linspace(10,200,1000)
dL = np.linspace(10,5000,500)

realistic_figaro = load_density("./density.pkl")
realistic_figaro = np.array([realistic_figaro[i].pdf(mz) for i in range(len(realistic_figaro))])

def realistic_jsd(x, i):
    z = CosmologicalParameters(np.float64(x[0]/100), 0.315, 0.685, -1, 0, 0).Redshift(dL)
    m = np.einsum("i, j -> ij", mz, np.reciprocal(1+z))
    
    p_realistic = np.einsum("ij, j -> ij", PLpeak(m, alpha=x[1]), DLsq(dL))
    p_realistic = np.trapz(p_realistic, dL, axis=1)
    p_realistic = p_realistic/np.trapz(p_realistic, mz, axis=0)

    return jsd(p_realistic, realistic_figaro[i])

def scipy_minimize(i):
    return minimize(realistic_jsd, x0=[np.random.uniform(20,60), np.random.uniform(-3,-1.1)], bounds = ((20,60),(-3,-1.1)), args=(i,)).x

with Pool(50) as p:
    result = p.map(scipy_minimize, range(len(realistic_figaro)))

np.save("./multi_dim_result.npy", result)
