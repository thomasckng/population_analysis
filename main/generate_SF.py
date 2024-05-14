import numpy as np
import dill

from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from cbc_pdet.o123_class_found_inj_general import Found_injections

# Selection function
run_fit = 'o3'
run_dataset = 'o3'
dmid_fun = 'Dmid_mchirp_expansion_noa30'
emax_fun = 'emax_exp'
alpha_vary = None

selfunc = Found_injections(dmid_fun, emax_fun, alpha_vary)
selfunc.load_inj_set(run_dataset)
selfunc.get_opt_params(run_fit)

n_pts  = np.array([200,200,200])

# Bounds
dl_bds  = [0.01,15000.]
m1_bds = [0.,500.]
q_bds  = [0.05, 1.]
bounds = np.array([m1_bds, q_bds, dl_bds])

# Arrays
m1    = np.linspace(m1_bds[0], m1_bds[1], n_pts[0])
q     = np.linspace(q_bds[0], q_bds[1], n_pts[1])
dl     = np.linspace(dl_bds[0], dl_bds[1], n_pts[2])
dn_m1 = np.prod(n_pts[1:])
dn_q  = np.prod(n_pts[2:])

# For loop
grid    = np.zeros(shape = (np.prod(n_pts), 3))
grid_dl = np.zeros(shape = (np.prod(n_pts), 3))
for i, m1i in tqdm(enumerate(m1), desc = 'Grid', total = n_pts[0]):
    for j, qi in enumerate(q):
        for k, dli in enumerate(dl):
            grid[i*dn_m1 + j*dn_q + k] = [m1i, qi, dli]

# Selection function in detector-frame mass
pdet = np.trapz(np.nan_to_num(selfunc.run_pdet(grid[:,2], grid[:,0], grid[:,1]*grid[:,0], 'o3').reshape(n_pts), nan = 0.), q, axis = 1)

selfunc_interp = RegularGridInterpolator((m1, dl), pdet, bounds_error = False, fill_value = 0.)

with open('selfunc_detframe.pkl', 'wb') as f:
    dill.dump(selfunc_interp, f)
