import numpy as np
import matplotlib.pyplot as plt

from corner import corner
from scipy.stats import norm
from distributions import *

from figaro.utils import rejection_sampler
from figaro.cosmology import CosmologicalParameters
from figaro.load import _find_redshift
from figaro import plot_settings

H0        = 40. # km/s/Mpc (Deliberately off value)
omega     = CosmologicalParameters(H0/100., 0.315, 0.685, -1., 0.)
n_draws   = 2000
norm_dist = norm()

# Simple distribution (two Gaussians)
DL = rejection_sampler(n_draws, gaussian, [0, 5000])
M  = np.array([mean(d) + std(d)*norm_dist.rvs() for d in DL])
z  = np.array([_find_redshift(omega, d) for d in DL])
Mz = M*(1+z)
np.savetxt('double_gaussian.txt', Mz)
# Plots
fig = corner(np.array([M, DL]).T,  labels = ['$\\mathrm{M}_1\ [\\mathrm{M}_\\odot]$', '$D_L\ [\\mathrm{Mpc}]$'])
fig.savefig('double_gaussian_corner.pdf', bbox_inches = 'tight')
fig, ax = plt.subplots()
ax.hist(Mz, histtype = 'step', density = True)
ax.set_xlabel('$\\mathrm{M}_z\ [\\mathrm{M}_\\odot]$')
ax.set_ylabel('$p(M_z)$')
fig.savefig('double_gaussian_detectorframe.pdf', bbox_inches = 'tight')

# Realistic distribution (DL^2, PL+Peak)
DL = rejection_sampler(n_draws, DLsq, [0, 5000])
M  = rejection_sampler(n_draws, PLpeak, [5,70])
z  = np.array([_find_redshift(omega, d) for d in DL])
Mz = M*(1+z)
np.savetxt('realistic.txt', Mz)
# Plots
fig = corner(np.array([M, DL]).T,  labels = ['$\\mathrm{M}_1\ [\\mathrm{M}_\\odot]$', '$D_L\ [\\mathrm{Mpc}]$'])
fig.savefig('realistic_corner.pdf', bbox_inches = 'tight')
fig, ax = plt.subplots()
ax.hist(Mz, histtype = 'step', density = True)
ax.set_xlabel('$M_z\ [\\mathrm{M}_\\odot]$')
ax.set_ylabel('$p(M_z)$')
fig.savefig('realistic_detectorframe.pdf', bbox_inches = 'tight')
