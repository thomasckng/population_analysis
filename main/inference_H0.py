import numpy as np
import matplotlib.pyplot as plt
from figaro.cosmology import CosmologicalParameters
from scipy.spatial.distance import jensenshannon as scipy_jsd
from figaro.load import load_density
import os
from figaro import plot_settings

# Mass distribution
from population_models.mass import plpeak

# Redshift distribution
def p_z(z, H0):
    return CosmologicalParameters(H0/100., 0.315, 0.685, -1., 0., 0.).ComovingVolumeElement(z)/(1+z)

label = "hierarchical_SE_test"
outdir = os.path.dirname(os.path.realpath(__file__)) + "/" + label

print("Preparing model pdfs...")
try:
    mz, H0, z, m, model_pdf = np.load(outdir+"/../model_pdf.npz").values()
except:
    mz = np.linspace(1,200,900)
    H0 = np.linspace(20,120,1000)
    z = np.linspace(0.001,2,800)
    m = np.einsum("i, j -> ij", mz, np.reciprocal(1+z)) # shape = (len(mz), len(z))

    # model mz pdf for each H0
    model_pdf = np.einsum("ij, kj -> ijk", plpeak(m), [p_z(z, i) for i in H0]) # shape = (len(mz), len(z), len(H0))

    from selection_function import selection_function
    from tqdm import tqdm
    SE_grid = np.array([selection_function(mz, CosmologicalParameters(i/100., 0.315, 0.685, -1., 0., 0.).LuminosityDistance(z).reshape(-1,1)) for i in tqdm(H0, desc = 'SE grid')]) # shape = (len(H0), len(mz), len(z))
    model_pdf = np.einsum("ijk, kji -> ijk", model_pdf, SE_grid) # shape = (len(mz), len(z), len(H0))

    model_pdf = np.trapz(model_pdf, z, axis=1) # shape = (len(mz), len(H0))

    np.savez(outdir+"/../model_pdf.npz", mz=mz, H0=H0, z=z, m=m, model_pdf=model_pdf)

print("Reading bounds and draws...")
draws = load_density(outdir+"/draws/draws_observed_"+label+".json")

bounds = np.loadtxt(outdir+"/jsd_bounds.txt")

print("Preparing H0 inference...")
# Mask out mz where there is no sample
_mask = [mz[k] <= bounds[1] and mz[k] >= bounds[0] for k in range(len(mz))]
mz_short = mz[_mask]
model_pdf_short = model_pdf[_mask]

figaro_pdf = np.array([draw.pdf(mz_short) for draw in draws])# shape (n_draws, len(mz_short))

from figaro.plot import plot_1d_dist

fig = plot_1d_dist(mz_short, figaro_pdf, save=False)
ax = fig.axes[0]

for i in range(0, len(H0), len(H0)//4):
    ax.plot(mz_short, model_pdf_short[:,i]/np.trapz(model_pdf_short[:,i], mz_short), label=f"H0={H0[i]:.1f}")
ax.legend()
fig.savefig(outdir+"/model_pdf.pdf")
fig.clf()
ax.clear()

print("Infering H0...")
# Compute JSD between (reconstructed observed distributions for each DPGMM draw) and (model mz distributions for each H0)
jsd = np.array([scipy_jsd(model_pdf_short, np.full((len(H0), len(mz_short)), figaro_pdf[j]).T) for j in range(len(figaro_pdf))])
# Find H0 that minimizes JSD for each DPGMM draw
H0_samples = H0[np.argmin(jsd, axis=1)]

print("Saving results...")
np.savetxt(outdir+"/jsds.txt", jsd)
np.savetxt(outdir+"/H0s.txt", H0_samples)

plt.hist(H0_samples)
percs = np.percentile(H0_samples, [5, 16, 50, 84, 95])
plt.axvline(70, lw = 0.7, ls = '--', c = 'orangered', label = '$\\mathrm{Simulated}$')
plt.axvline(percs[2], c = 'steelblue', lw=0.7, label = '$H_0 = '+str(f'{percs[2]:.1f}')+'^{+'+str(f'{percs[3]-percs[2]:.1f}')+'}_{-'+str(f'{percs[2]-percs[1]:.1f}')+'}$')
plt.axvspan(percs[1], percs[3], alpha=0.25, color='mediumturquoise')
plt.axvspan(percs[0], percs[4], alpha=0.25, color='darkturquoise')
plt.savefig(outdir+"/H0_hist.pdf")

print("Done!")
