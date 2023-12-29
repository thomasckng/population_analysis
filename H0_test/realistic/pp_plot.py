import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon as scipy_jsd
from figaro.mixture import DPGMM
from figaro.utils import get_priors
from scipy.stats import norm
from figaro.utils import rejection_sampler
from figaro.cosmology import CosmologicalParameters
from figaro.load import _find_redshift
from figaro import plot_settings
# import ray

# ray.init()

# Mass distribution
def PLpeak(m, alpha = -2., mmin = 5., mmax = 70., mu = 30., sigma = 4., w = 0.2):
    norm_pl = (1 - alpha) / (mmin ** (alpha + 1) - mmax ** (alpha + 1))
    pl      = norm_pl * m ** alpha
    peak    = np.exp(-0.5 * ((m - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    return w * pl + (1 - w) * peak

# dL distribution
def dLsq(dL, dLmax = 5000):
    return 3 * dL ** 2 / dLmax ** 3

true_H0 = 40. # km/s/Mpc (Deliberately off value)
omega = CosmologicalParameters(true_H0/100., 0.315, 0.685, -1., 0.)
n_draws_samples = 2000
norm_dist = norm()

# Generate samples from source distribution
print("Generating samples...")
M_min = 0
M_max = 200

Mz_list = []
for _ in tqdm(range(100)):
    in_bound = False
    while not in_bound:
        dL = rejection_sampler(n_draws_samples, dLsq, [0, 5000])
        M  = rejection_sampler(n_draws_samples, PLpeak, [5,70])
        z  = np.array([_find_redshift(omega, d) for d in dL])
        Mz = M * (1 + z)
        in_bound = Mz.max() < M_max
    Mz_list.append(Mz)
Mz_list = np.array(Mz_list)

print("Preparing model pdfs...")
mz = np.linspace(10,200,1000)
dL = np.linspace(10,5000,500)

H0 = np.linspace(20,60,500)

def luminosity_distance_to_redshift(distance, H0s):
    return np.array([_find_redshift(CosmologicalParameters(H0/100, 0.315, 0.685, -1, 0), distance) for H0 in H0s])

z = luminosity_distance_to_redshift(dL, H0)

m = np.einsum("i, jk -> ikj", mz, np.reciprocal(1+z))

# model mz pdf for each H0
model_pdf = np.trapz(np.einsum("ijk, j -> ijk", PLpeak(m), dLsq(dL)), dL, axis=1)
model_pdf = model_pdf / np.trapz(model_pdf, mz, axis=0)

# @ray.remote
# def reconstruct_observed_distribution(samples):
#     samples = Mz_list[i]
#     mix = DPGMM([[M_min, M_max]], prior_pars=get_priors([[M_min, M_max]], samples))
#     return mix.density_from_samples(samples)

# @ray.remote
# def compute_jsd(i):
#     return scipy_jsd(model_pdf, np.full((len(H0), len(mz)), figaro_pdf[i]).T)

print("Computing pp plot...")
n_draws_figaro = 1000

H0_perc = []
for i in tqdm(range(100)):
    # Reconstruct observed distribution using FIGARO
    samples = Mz_list[i]
    mix_double_gaussian = DPGMM([[M_min, M_max]], prior_pars=get_priors([[M_min, M_max]], samples))
    draws = np.array([mix_double_gaussian.density_from_samples(samples) for _ in range(n_draws_figaro)])
    # draws = [reconstruct_observed_distribution.remote(i) for _ in range(n_draws_figaro)]
    # draws = np.array(ray.get(draws))

    figaro_pdf = np.array([draw.pdf(mz) for draw in draws])

    # Compute JSD between (reconstructed observed distributions for each DPGMM draw) and (model mz distributions for each H0)
    jsd = np.array([scipy_jsd(model_pdf, np.full((len(H0), len(mz)), figaro_pdf[j]).T) for j in range(len(figaro_pdf))])
    # jsd = [compute_jsd.remote(j) for j in range(len(figaro_pdf))]
    # jsd = np.array(ray.get(jsd))

    # Find H0 that minimizes JSD for each DPGMM draw
    H0_samples = H0[np.argmin(jsd, axis=1)]
    
    # Compute percentage of H0 samples that are smaller than true H0
    H0_perc.append(np.sum(H0_samples<=true_H0) / len(H0_samples))
H0_perc = np.array(H0_perc)
np.save("H0_perc.npy", H0_perc)

# pp plot
print("Plotting...")
plt.figure(figsize=(6,6))
plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), "k--")
plt.plot(np.sort(H0_perc), np.linspace(0,1,100), "r-")
plt.savefig("pp_plot.pdf")
