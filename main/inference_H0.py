import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon as scipy_jsd
from figaro.cosmology import CosmologicalParameters
from figaro.load import load_density
from figaro.marginal import marginalise
from multiprocessing import Pool

from population_models.mass import plpeak

outdir = "/Users/thomas.ng/Documents/GitHub/population_analysis/main/hierarchical_SE_test_3"

print("Preparing model pdfs...")
mz = np.linspace(0,200,1000)
H0 = np.linspace(20,120,1000)
z = np.linspace(0,1,1000)
m = np.einsum("i, j -> ij", mz, np.reciprocal(1+z))

def dVdz(H0):
    return CosmologicalParameters(H0/100., 0.315, 0.685, -1., 0., 0.).ComovingVolumeElement(z)

# model mz pdf for each H0
model_pdf = np.sum(np.einsum("ij, kj -> ijk", plpeak(m), np.einsum("ij, j -> ij", np.array([dVdz(i) for i in H0]), np.exp(-z/0.1)*np.reciprocal(1+z)))*(z[1]-z[0]), axis=1)

print("Reading draws...")

draws = load_density(outdir+"/draws/draws_intrinsic_hierarchical_SE_test_3.json")

print("Infering H0...")

bounds = np.loadtxt(outdir+"/jsd_bounds.txt")
bounds = np.atleast_2d([[10,80]])

# Mask out mz where there is no sample
mask = [mz[k] <= bounds[0,1] and mz[k] >= bounds[0,0] for k in range(len(mz))]
mz_short = mz[mask]
model_pdf_short = model_pdf[mask]

mass_draws = marginalise(draws, [1])
figaro_pdf = np.array([draw.pdf(mz_short) for draw in mass_draws])

# Compute JSD between (reconstructed observed distributions for each DPGMM draw) and (model mz distributions for each H0)
jsd = np.array([scipy_jsd(model_pdf_short, np.full((len(H0), len(mz_short)), figaro_pdf[j]).T) for j in range(len(figaro_pdf))])
# Find H0 that minimizes JSD for each DPGMM draw
H0_samples = H0[np.argmin(jsd, axis=1)]

print("Saving results...")

np.savetxt(outdir+"/jsds.txt", jsd)
np.savetxt(outdir+"/H0s.txt", H0_samples)

plt.hist(H0_samples)
plt.savefig(outdir+"/H0_hist.pdf")

print("Done!")
