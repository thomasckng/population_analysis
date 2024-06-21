import numpy as np
from numpy.random import uniform as uni
from scipy.spatial.distance import jensenshannon as scipy_jsd
from figaro.load import load_density
from figaro.cosmology import CosmologicalParameters
from multiprocessing import Pool
from selection_function import selection_function
from population_models.mass import plpeak
import sys
import os

# Redshift distribution
def p_z(z, H0):
    return CosmologicalParameters(H0/100., 0.315, 0.685, -1., 0., 0.).ComovingVolumeElement(z)/(1+z)

# Jacobian
def jacobian(func):
    def wrapper(z, *args):
        return func(z, *args)/(1+z)
    return wrapper

if len(sys.argv) < 4:
    print("Invalid number of arguments!")
    sys.exit(1)
elif sys.argv[1] == "alpha":
    x0 = [uni(10,200), uni(1.01,5)]
    bounds = ((10,200), (1.01,5))
    def plp(m, x):
        return plpeak(m, alpha=x[0])
elif sys.argv[1] == "mu":
    x0 = [uni(10,200), uni(10,50)]
    bounds = ((10,200), (10,50))
    def plp(m, x):
        return plpeak(m, mu=x[0])
elif sys.argv[1] == "4":
    x0 = [uni(10,200), uni(10,50), uni(0.01,10), uni(0.01,15)]
    bounds = ((10,200), (10,50), (0.01,10), (0.01,15))
    def plp(m, x):
        return plpeak(m, mu=x[0], sigma=x[1], delta=x[2])
elif sys.argv[1] == "5":
    x0 = [uni(10,200), uni(1.01,5), uni(10,50), uni(0.01,10), uni(0,1)]
    bounds = ((10,200), (1.01,5), (10,50), (0.01,10), (0,1))
    def plp(m, x):
        return plpeak(m, alpha=x[0], mu=x[1], sigma=x[2], w=x[3])
elif sys.argv[1] == "5a":
    x0 = [uni(10,200), uni(1.01,5), uni(10,50), uni(1,10), uni(0,1)]
    bounds = ((10,200), (1.01,5), (10,50), (1,10), (0,1))
    def plp(m, x):
        return plpeak(m, alpha=x[0], mu=x[1], mmin=x[2], w=x[3])
elif sys.argv[1] == "6":
    x0 = [uni(10,200), uni(1.01,5), uni(10,50), uni(0.01,10), uni(0,1), uni(0.01,15)]
    bounds = ((10,200), (1.01,5), (10,50), (0.01,10), (0,1), (0.01,15))
    def plp(m, x):
        return plpeak(m, alpha=x[0], mu=x[1], sigma=x[2], w=x[3], delta=x[4])
else:
    print("Invalid argument!")
    sys.exit(1)

def jsd(x, i):
    model_pdf = np.einsum("ij, j -> ij", plp(m, x[1:]), jacobian(p_z)(z, x[0])) # shape = (len(mz), len(z))
    grid = np.transpose(np.meshgrid(mz, CosmologicalParameters(x[0]/100., 0.315, 0.685, -1., 0., 0.).LuminosityDistance(z))) # shape = (len(mz), len(z), 2)
    SE_grid = selection_function(grid) # shape = (len(mz), len(z))
    model_pdf = np.einsum("ij, ij -> ij", model_pdf, SE_grid) # shape = (len(mz), len(z))
    model_pdf = np.trapz(model_pdf, z, axis=1) # shape = (len(mz))
    model_pdf_short = model_pdf[_mask]

    return scipy_jsd(model_pdf_short, pdf_figaro[i])


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


n_pool = 32

label = sys.argv[3]
outdir = os.path.dirname(os.path.realpath(__file__)) + "/" + label

mz = np.linspace(1,200,900)
H0 = np.linspace(5,150,1000)
z = np.linspace(0.001,2,800)
m = np.einsum("i, j -> ij", mz, np.reciprocal(1+z)) # shape = (len(mz), len(z))

print("Reading bounds and draws...")
draws = load_density(outdir+"/draws/draws_observed_"+label+".json")

jsd_bounds = np.loadtxt(outdir+"/jsd_bounds.txt")

print("Preparing inference...")
# Mask out mz where there is no sample
_mask = [mz[k] <= jsd_bounds[1] and mz[k] >= jsd_bounds[0] for k in range(len(mz))]
mz_short = mz[_mask]

pdf_figaro = np.array([draw.pdf(mz_short) for draw in draws])# shape (n_draws, len(mz_short))

print("Minimizing JSD...")
with Pool(n_pool) as p:
    result = p.map(minimize, range(len(pdf_figaro)))

print("Saving results...")
if not os.path.exists(outdir+'/multi'):
    os.makedirs(outdir+'/multi')
np.savez(outdir+"/multi/"+sys.argv[1]+"_"+sys.argv[2]+".npz", result=result, pdf_figaro=pdf_figaro)
print("Done!")
