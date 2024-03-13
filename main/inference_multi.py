import numpy as np
from numpy.random import uniform as uni
from scipy.spatial.distance import jensenshannon as scipy_jsd
from figaro.mixture import DPGMM
from figaro.utils import rejection_sampler, get_priors
from figaro.cosmology import CosmologicalParameters
from multiprocessing import Pool
from population_models.mass import plpeak
import sys

# dL distributions
def DLsq(DL, DLmax = 5000):
    return 3*DL**2/DLmax**3


def reconstruct_observed_distribution(samples):
    mix = DPGMM([[M_min, M_max]], prior_pars=get_priors([[M_min, M_max]], samples))
    return mix.density_from_samples(samples)


if len(sys.argv) < 3:
    print("Invalid number of arguments!")
    sys.exit(1)
elif sys.argv[1] == "alpha":
    x0 = [uni(10,100), uni(1.01,5)]
    bounds = ((10,100), (1.01,5))
    def plp(m, x):
        return plpeak(m, alpha=x[0])
elif sys.argv[1] == "mu":
    x0 = [uni(10,100), uni(10,50)]
    bounds = ((10,100), (10,50))
    def plp(m, x):
        return plpeak(m, mu=x[0])
elif sys.argv[1] == "5":
    x0 = [uni(10,100), uni(1.01,5), uni(10,50), uni(0.01,10), uni(0,1)]
    bounds = ((10,100), (1.01,5), (10,50), (0.01,10), (0,1))
    def plp(m, x):
        return plpeak(m, alpha=x[0], mu=x[1], sigma=x[2], w=x[3])
elif sys.argv[1] == "8":
    x0 = [uni(10,100), uni(1.01,5), uni(10,50), uni(0.01,10), uni(0,1), uni(0.01,10), uni(50,200), uni(0.01,10)]
    bounds = ((10,100), (1.01,5), (10,50), (0.01,10), (0,1), (0.01,10), (50,200), (0.01,10))
    def plp(m, x):
        return plpeak(m, alpha=x[0], mu=x[1], sigma=x[2], w=x[3], mmin=x[4], mmax=x[5], delta=x[6])
else:
    print("Invalid argument!")
    sys.exit(1)

def jsd(x, i):
        z = CosmologicalParameters(x[0]/100., 0.315, 0.685, -1., 0., 0.).Redshift(dL)
        m = np.einsum("i, j -> ij", mz, np.reciprocal(1+z))
        
        p = np.einsum("ij, j -> ij", plp(m, x[1:]), DLsq(dL))
        p = np.sum(p, axis=1) # delta_dL and normalization of mz is included in scipy_jsd

        return scipy_jsd(p, pdf_figaro[i])


if sys.argv[2] in ["Nelder-Mead", "Powell", "L-BFGS-B", "TNC", "COBYLA", "SLSQP", "trust-constr"]:

    method = sys.argv[2]

    from scipy.optimize import minimize as scipy_minimize

    def minimize(i):
        return scipy_minimize(jsd, x0=x0, bounds=bounds, args=(i,), method=method).x
    
elif sys.argv[2] == "CMA-ES":

    import cma

    def minimize(i):
        return cma.fmin2(jsd, x0, 1, {'bounds': np.array(bounds).T.tolist(), 'CMA_stds': np.array(bounds).T[1]/4}, args=(i,))[0]
    
elif sys.argv[2] == "PSO":

    def jsd_ps(x, i):
        return np.array([jsd(x[j], i) for j in range(len(x))])

    import pyswarms as ps

    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}
    optimizer = ps.single.LocalBestPSO(n_particles=10, dimensions=len(x0), options=options, bounds=tuple(np.array(bounds).T))

    def minimize(i):
        return optimizer.optimize(jsd_ps, iters=100, i=i)[1]
    
else:
    print("Invalid argument!")
    sys.exit(1)


n_pool = 16

print("Generating samples from source distribution...")
true_H0 = 70. # km/s/Mpc

n_draws_samples = 1000
n_draws_figaro = 10000
M_min = 0
M_max = 200

i = 0
while i < 10:
    try:
        # Generate samples from source distribution
        valid = False
        while not valid:
            dL_sample = rejection_sampler(n_draws_samples, DLsq, [1,5000])
            M_sample  = rejection_sampler(n_draws_samples, plpeak, [1,200])
            z_sample  = CosmologicalParameters(true_H0/100., 0.315, 0.685, -1., 0., 0.).Redshift(dL_sample)
            Mz_sample = M_sample * (1 + z_sample)
            valid = Mz_sample.max() < M_max and Mz_sample.min() > M_min

        print("Reconstructing observed distribution...")
        mz = np.linspace(np.min(Mz_sample),np.max(Mz_sample),1000)
        dL = np.linspace(10,5000,1000)

        with Pool(n_pool) as p:
            pdf_figaro = p.map(reconstruct_observed_distribution, np.full((n_draws_figaro,len(Mz_sample)), Mz_sample))
        pdf_figaro = np.array([pdf_figaro[i].pdf(mz) for i in range(len(pdf_figaro))])
    except Exception as e:
        i = i + 1
        print("An exception occurred:", e)
        continue # skip remaining code and try again
    break # break while loop
else:
    print("Failed for 10 times")
    sys.exit()

print("Minimizing JSD...")
with Pool(n_pool) as p:
    result = p.map(minimize, range(len(pdf_figaro)))

print("Saving results...")
np.savez("./result/result_"+sys.argv[1]+"_"+sys.argv[2]+".npz", result=result, Mz_sample=Mz_sample, pdf_figaro=pdf_figaro)
print("Done!")