import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon as scipy_jsd
from figaro.mixture import DPGMM
from figaro.utils import get_priors, rejection_sampler
from figaro.cosmology import CosmologicalParameters
from multiprocessing import Pool
import sys
import pickle
# from numba import njit
from population_models.mass import plpeak

def reconstruct_observed_distribution(samples):
    mix = DPGMM([[M_min, M_max]], prior_pars=get_priors([[M_min, M_max]], samples))
    return mix.density_from_samples(samples)

if __name__ == '__main__':
    # dL distribution
    # @njit
    def dLsq(dL, dLmax = 5000):
        return 3 * dL ** 2 / dLmax ** 3
    
    n_runs = 100
    
    true_H0 = np.random.uniform(40, 100, n_runs)
    true_omega = [CosmologicalParameters(true_H0[i]/100., 0.315, 0.685, -1., 0., 0.) for i in range(n_runs)]

    print("Preparing model pdfs...")
    mz = np.linspace(1,200,2000)
    dL = np.linspace(1,5000,500)

    H0 = np.linspace(20,60,500)

    def luminosity_distance_to_redshift(distance, H0s):
        return np.array([CosmologicalParameters(H0/100., 0.315, 0.685, -1., 0., 0.).Redshift(distance) for H0 in H0s])

    z = luminosity_distance_to_redshift(dL, H0)

    m = np.einsum("i, jk -> ikj", mz, np.reciprocal(1+z))

    # model mz pdf for each H0
    model_pdf = np.trapz(np.einsum("ijk, j -> ijk", plpeak(m), dLsq(dL)), dL, axis=1)
    model_pdf = model_pdf / np.trapz(model_pdf, mz, axis=0)

    print("Computing pp plot...")
    n_draws_samples = 2000
    n_draws_figaro = 1000
    M_min = 0
    M_max = 200

    figaro_pdf_arr = []
    jsd_arr = []
    H0_samples_arr = []
    H0_perc_arr = []
    with Pool(64) as p:
        for i in tqdm(range(n_runs)):
            j = 0
            while j < 10:
                try:
                    # Generate samples from source distribution
                    valid = False
                    while not valid:
                        dL_sample = rejection_sampler(n_draws_samples, dLsq, [1,5000])
                        M_sample  = rejection_sampler(n_draws_samples, plpeak, [1,200])
                        z_sample  = np.array([true_omega[i].Redshift(d) for d in dL_sample])
                        Mz_sample = M_sample * (1 + z_sample)
                        valid = Mz_sample.max() < M_max and Mz_sample.min() > M_min

                    # Reconstruct observed distribution using FIGARO
                    draws = p.map(reconstruct_observed_distribution, np.full((n_draws_figaro,len(Mz_sample)), Mz_sample))
                except Exception as e:
                    j = j + 1
                    print("An exception occurred:", e)
                    continue # skip remaining code and try again

                # Mask out mz where there is no sample
                mask = [mz[k] <= np.max(Mz_sample) and mz[k] >= np.min(Mz_sample) for k in range(len(mz))]
                mz_short = mz[mask]
                model_pdf_short = model_pdf[mask]

                figaro_pdf = np.array([draw.pdf(mz_short) for draw in draws])
                figaro_pdf_arr.append(figaro_pdf)

                # Compute JSD between (reconstructed observed distributions for each DPGMM draw) and (model mz distributions for each H0)
                jsd = np.array([scipy_jsd(model_pdf_short, np.full((len(H0), len(mz_short)), figaro_pdf[j]).T) for j in range(len(figaro_pdf))])
                jsd_arr.append(jsd)

                # Find H0 that minimizes JSD for each DPGMM draw
                H0_samples = H0[np.argmin(jsd, axis=1)]
                H0_samples_arr.append(H0_samples)
                
                # Compute percentage of H0 samples that are smaller than true H0
                H0_perc = np.sum(H0_samples<=true_H0[i]) / len(H0_samples)
                H0_perc_arr.append(H0_perc)
                break # break while loop
            else:
                print("Failed for 10 times")
                sys.exit()

    with open("figaro_pdf_arr.pkl", "wb") as f:
        pickle.dump(figaro_pdf_arr, f)
    jsd_arr = np.array(jsd_arr)
    H0_samples_arr = np.array(H0_samples_arr)
    H0_perc_arr = np.array(H0_perc_arr)
    np.savez("result_pp_plot.npz", true_H0=true_H0, jsd_arr=jsd_arr, H0_samples_arr=H0_samples_arr, H0_perc_arr=H0_perc_arr)
