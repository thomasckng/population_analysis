import numpy as np
import matplotlib.pyplot as plt
from figaro.cosmology import CosmologicalParameters
from scipy.spatial.distance import jensenshannon as scipy_jsd
from figaro.load import load_density
import os
from figaro.plot import plot_1d_dist, plot_median_cr
from figaro import plot_settings
import sys

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Mass distribution
from population_models.mass import plpeak

# Redshift distribution
def p_z(z, H0):
    return CosmologicalParameters(H0/100., 0.315, 0.685, -1., 0., 0.).ComovingVolumeElement(z)/(1+z)

# Jacobian
def jacobian(func):
    def wrapper(z, *args):
        return func(z, *args)/(1+z)
    return wrapper

label = sys.argv[1]
outdir = os.path.dirname(os.path.realpath(__file__)) + "/" + label

print("Preparing model pdfs...")
try:
    mz, H0, z, m, model_pdf = np.load(outdir+"/../real_model_pdf.npz").values()
except:
    mz = np.linspace(1,200,900)
    H0 = np.linspace(5,150,1000)
    z = np.linspace(0.001,2,800)
    m = np.einsum("i, j -> ij", mz, np.reciprocal(1+z)) # shape = (len(mz), len(z))

    # model mz pdf for each H0
    model_pdf = np.einsum("ij, kj -> ijk", plpeak(m), [jacobian(p_z)(z, i) for i in H0]) # shape = (len(mz), len(z), len(H0))

    from selection_function import selection_function
    from tqdm import tqdm
    grid = [np.transpose(np.meshgrid(mz, CosmologicalParameters(i/100., 0.315, 0.685, -1., 0., 0.).LuminosityDistance(z))) for i in H0] # shape = (len(H0), len(mz), len(z), 2)
    SE_grid = np.array([selection_function(grid[i]) for i in tqdm(range(len(H0)), desc = 'SE grid')]) # shape = (len(H0), len(mz), len(z))
    model_pdf = np.einsum("ijk, kij -> ijk", model_pdf, SE_grid) # shape = (len(mz), len(z), len(H0))

    model_pdf = np.trapz(model_pdf, z, axis=1) # shape = (len(mz), len(H0))

    np.savez(outdir+"/../real_model_pdf.npz", mz=mz, H0=H0, z=z, m=m, model_pdf=model_pdf)

print("Reading bounds and draws...")
draws = load_density(outdir+"/draws/draws_observed_"+label+".json")

bounds = np.loadtxt(outdir+"/jsd_bounds.txt")

print("Preparing H0 inference...")
# Mask out mz where there is no sample
_mask = [mz[k] <= bounds[1] and mz[k] >= bounds[0] for k in range(len(mz))]
mz_short = mz[_mask]
model_pdf_short = model_pdf[_mask]

figaro_pdf = np.array([draw.pdf(mz_short) for draw in draws])# shape (n_draws, len(mz_short))

print("Infering H0...")
# Compute JSD between (reconstructed observed distributions for each DPGMM draw) and (model mz distributions for each H0)
jsd = np.array([scipy_jsd(model_pdf_short, np.full((len(H0), len(mz_short)), figaro_pdf[j]).T) for j in range(len(figaro_pdf))])
# Find H0 that minimizes JSD for each DPGMM draw
H0_samples = H0[np.argmin(jsd, axis=1)]
H0_samples = H0_samples[H0_samples < 140]

print("Saving results...")
np.savetxt(outdir+"/jsds.txt", jsd)
np.savetxt(outdir+"/H0s.txt", H0_samples)

print("Plotting results...")

# obs_samples = np.loadtxt(outdir+'/obs_samples.txt')
# plt.hist(obs_samples, bins = int(np.sqrt(len(obs_samples))), histtype = 'step', density = True, label = '$\mathrm{Samples}$')
# plt.xlabel('$M_z\ [\mathrm{M}_\odot]$')
# plt.ylabel('Density')
# plt.xlim(0,120)
# plt.ylim(0,0.05)
# plt.title('Observed detector-frame mass distribution', fontsize=20)
# plt.legend()
# plt.savefig(outdir+'/observed_mass.pdf', bbox_inches='tight')
# plt.clf()

# fig = plot_median_cr(draws, samples=obs_samples, save=True, show=False)
# ax = fig.axes[0]
# ax.set_xlabel('$M_z\ [\mathrm{M}_\odot]$')
# ax.set_ylabel('$p(M_z|\Theta_i(\mathbf{Y}))$')
# ax.set_xlim(0,120)
# ax.set_ylim(0,0.05)
# ax.set_title('Reconstructed detector-frame mass distribution', fontsize=20)
# fig.savefig(outdir+'/observed_figaro.pdf', bbox_inches='tight')
# fig.clf()
# ax.clear()
# plt.clf()

# m = np.linspace(0,120,1000)
# plt.plot(m, plpeak(m)/np.trapz(plpeak(m), m), label = "$\mathrm{PL+Peak}$")
# plt.xlabel('$M\ [\mathrm{M}_\odot]$')
# plt.ylabel('$p(M|\Lambda)$')
# plt.legend()
# plt.xlim(0,120)
# plt.ylim(0)
# plt.title("Source-frame mass distribution",  fontsize=20)
# plt.savefig(outdir+"/model_source_frame.pdf", bbox_inches='tight')
# plt.clf()

colors = ['tab:green', 'tab:red', 'tab:orange']
# for i, c in zip([np.argmin(abs(H0-40)), np.argmin(abs(H0-70)), np.argmin(abs(H0-100))], colors):
#     plt.plot(mz, model_pdf[:,i]/np.trapz(model_pdf[:,i], mz), label=f"$H_0={H0[i]:.0f}$", c=c)
# plt.legend()
# plt.xlabel('$M_z\ [\mathrm{M}_\odot]$')
# plt.ylabel('$p(M_z|\Lambda, \Omega)$')
# plt.xlim(0,120)
# plt.ylim(0,0.05)
# plt.title("Detector-frame mass distribution",  fontsize=20)
# plt.savefig(outdir+"/model_detector_frame.pdf", bbox_inches='tight')
# plt.clf()

fig = plot_1d_dist(mz_short, figaro_pdf, save=False)
ax = fig.axes[0]
for i, c in zip([np.argmin(abs(H0-40)), np.argmin(abs(H0-70)), np.argmin(abs(H0-100))], colors):
    ax.plot(mz_short, model_pdf_short[:,i]/np.trapz(model_pdf_short[:,i], mz_short), label=f"$H_0={H0[i]:.0f}$", c=c)
ax.legend()
ax.set_xlabel('$M_z\ [\mathrm{M}_\odot]$')
ax.set_ylabel('Density')
ax.set_xlim(bounds[0],bounds[1])
ax.set_ylim(0,0.05)
fig.savefig(outdir+"/Comparison.pdf")
fig.clf()
ax.clear()
plt.clf()

plt.hist(H0_samples)
percs = np.percentile(H0_samples, [5, 16, 50, 84, 95])
# plt.axvline(70, lw = 0.7, ls = '--', c = 'orangered', label = '$\\mathrm{Simulated}$')
plt.axvline(percs[2], c = 'steelblue', lw=0.7, label = '$H_0 = '+str(f'{percs[2]:.1f}')+'^{+'+str(f'{percs[3]-percs[2]:.1f}')+'}_{-'+str(f'{percs[2]-percs[1]:.1f}')+'}$')
plt.axvspan(percs[1], percs[3], alpha=0.25, color='mediumturquoise')
plt.axvspan(percs[0], percs[4], alpha=0.25, color='darkturquoise')
# plt.xlim(20,120)
plt.legend()
plt.xlabel('$H_0\ [\mathrm{km/s/Mpc}]$')
plt.ylabel('Density')
plt.savefig(outdir+"/H0_inference.pdf", bbox_inches='tight')
plt.clf()

print("Done!")
