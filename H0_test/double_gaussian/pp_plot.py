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
from multiprocessing import Pool
import sys

def reconstruct_observed_distribution(samples):
    mix = DPGMM([[M_min, M_max]], prior_pars=get_priors([[M_min, M_max]], samples))
    return mix.density_from_samples(samples)

if __name__ == '__main__':
    # Mass distribution
    def mean(dL, dMddL = 30./1000., offset = 5.):
        return dL*dMddL + offset
    
    def std(dL, dMddL = 8./1000., offset = 1.):
        return dL*dMddL + offset
    
    def evolving_gaussian(m, dL):
        mu  = mean(dL)
        sigma = std(dL)
        return np.exp(-0.5 * ((m - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)

    # dL distribution
    def gaussian(dL, mu = 1600, sigma = 400):
        return np.exp(-0.5 * ((dL - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)

    true_H0 = 40. # km/s/Mpc (Deliberately off value)
    omega = CosmologicalParameters(true_H0/100., 0.315, 0.685, -1., 0.)
    n_draws_samples = 2000
    norm_dist = norm()

    print("Preparing model pdfs...")
    mz = np.linspace(10,200,1000)
    dL = np.linspace(10,5000,500)

    H0 = np.linspace(20,60,500)

    def luminosity_distance_to_redshift(distance, H0s):
        return np.array([_find_redshift(CosmologicalParameters(H0/100, 0.315, 0.685, -1, 0), distance) for H0 in H0s])

    z = luminosity_distance_to_redshift(dL, H0)

    m = np.einsum("i, jk -> ikj", mz, np.reciprocal(1+z))

    # model mz pdf for each H0
    model_pdf = np.trapz(np.einsum("ijk, j -> ijk", evolving_gaussian(m, dL.reshape(-1,1)), gaussian(dL)), dL, axis=1)
    model_pdf = model_pdf / np.trapz(model_pdf, mz, axis=0)

    print("Computing pp plot...")
    n_draws_figaro = 1000
    M_min = 0
    M_max = 200

    figaro_pdf_arr = []
    jsd_arr = []
    H0_samples_arr = []
    H0_perc_arr = []
    with Pool(50) as p:
        for i in tqdm(range(100)):
            i = 0
            while i < 10:
                try:
                    # Generate samples from source distribution
                    valid = False
                    while not valid:
                        dL = rejection_sampler(n_draws_samples, gaussian, [0, 5000])
                        M  = np.array([mean(d) + std(d) * norm_dist.rvs() for d in dL])
                        z  = np.array([_find_redshift(omega, d) for d in dL])
                        Mz = M * (1 + z)
                        valid = Mz.max() < M_max and not np.sum(np.isnan(Mz), dtype = 'bool')

                    # Reconstruct observed distribution using FIGARO
                    draws = p.map(reconstruct_observed_distribution, np.full((n_draws_figaro,len(Mz)), Mz))
                except Exception as e:
                    i = i + 1
                    print("An exception occurred:", e)
                    continue # skip remaining code and try again

                figaro_pdf = np.array([draw.pdf(mz) for draw in draws])
                figaro_pdf_arr.append(figaro_pdf)

                # Compute JSD between (reconstructed observed distributions for each DPGMM draw) and (model mz distributions for each H0)
                jsd = np.array([scipy_jsd(model_pdf, np.full((len(H0), len(mz)), figaro_pdf[j]).T) for j in range(len(figaro_pdf))])
                jsd_arr.append(jsd)

                # Find H0 that minimizes JSD for each DPGMM draw
                H0_samples = H0[np.argmin(jsd, axis=1)]
                H0_samples_arr.append(H0_samples)
                
                # Compute percentage of H0 samples that are smaller than true H0
                H0_perc = np.sum(H0_samples<=true_H0) / len(H0_samples)
                H0_perc_arr.append(H0_perc)
                break # break while loop
            else:
                print("Failed for 10 times")
                sys.exit()

    figaro_pdf_arr = np.array(figaro_pdf_arr)
    jsd_arr = np.array(jsd_arr)
    H0_samples_arr = np.array(H0_samples_arr)
    H0_perc_arr = np.array(H0_perc_arr)
    np.savez("result.npz", figaro_pdf_arr, jsd_arr, H0_samples_arr, H0_perc_arr)

    # pp plot
    print("Plotting...")
    plt.figure(figsize=(6,6))
    plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), "k--")
    plt.plot(np.sort(H0_perc_arr), np.linspace(0,1,100), "r-")
    plt.savefig("pp_plot.pdf")
