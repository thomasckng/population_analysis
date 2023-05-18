import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".7"

import numpy as np
import jax
import jax.numpy as jnp
from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys

z_0 = 2
s_z = 0.3
M_0 = 50
sigma_M = 5
true_param = [z_0, s_z, M_0, sigma_M]
n_param = len(true_param)

N = 100
M = np.random.normal(M_0, sigma_M, N)
z = np.random.normal(z_0, s_z, N)
M_z = M*(1+z)
print("Mean of M: ", M.mean())

def log_p_z(z, z_0, sigma_z):
    return -jnp.log(sigma_z)-0.5*jnp.log(2*np.pi)-0.5*((z-z_0)/sigma_z)**2

def log_p_M_z(M_z, z, M_0, sigma_M):
    return -jnp.log(sigma_M)-0.5*jnp.log(2*np.pi)-0.5*((M_z/(1+z)-M_0)/sigma_M)**2

def log_likelihood(x):
    return jnp.sum(log_p_M_z(M_z, x[n_param:], x[2], x[3]) + log_p_z(x[n_param:], x[0], x[1]) - jnp.log(1+x[n_param:]))

n_dim = n_param+N
n_chains = 1000
n_loop_training = 20
n_loop_production = 20
n_local_steps = 500
n_global_steps = 500
learning_rate = 0.001
max_samples = 100000
momentum = 0.9
num_epochs = 60
batch_size = 50000

rng_key_set = initialize_rng_keys(n_chains, seed=42)

prior_range = jnp.array([[0,5],[0,3],[0,100],[0,10],[0,10]])

def prior(x):
    output = 0.
    for i in range(n_param):
        output = jax.lax.cond(x[i]>=prior_range[i,0], lambda: output, lambda: -jnp.inf)
        output = jax.lax.cond(x[i]<=prior_range[i,1], lambda: output, lambda: -jnp.inf)
    for i in range(n_param,n_dim):
        output = jax.lax.cond(x[i]>=prior_range[n_param,0], lambda: output, lambda: -jnp.inf)
        output = jax.lax.cond(x[i]<=prior_range[n_param,1], lambda: output, lambda: -jnp.inf)
    return output

def posterior(x):
    return log_likelihood(x) + prior(x)

initial_position = jax.random.uniform(rng_key_set[0], shape=(int(n_chains), n_dim)) * 1
for i in range(n_param):
    initial_position = initial_position.at[:,i].set(initial_position[:,i]*(prior_range[i,1]-prior_range[i,0])+prior_range[i,0])
for i in range(n_param,n_dim):
    initial_position = initial_position.at[:,i].set(initial_position[:,i]*(prior_range[n_param,1]-prior_range[n_param,0])+prior_range[n_param,0])

model = RQSpline(n_dim, 10, [128, 128], 8)

step_size = 1e-1
local_sampler = MALA(log_likelihood, True, {"step_size": step_size})

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    local_sampler,
    posterior,
    model,
    n_loop_training=n_loop_training,
    n_loop_production = n_loop_production,
    n_local_steps=n_local_steps,
    n_global_steps=n_global_steps,
    n_chains=n_chains,
    n_epochs=num_epochs,
    learning_rate=learning_rate,
    momentum=momentum,
    batch_size=batch_size,
    use_global=True,
    keep_quantile=0.,
    train_thinning = 40
)

nf_sampler.sample(initial_position)
chains,log_prob,local_accs, global_accs = nf_sampler.get_sampler_state().values()

np.savez("./result.npz", chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)

print("Local acceptance rate: ", np.mean(local_accs))
print("Global acceptance rate: ", np.mean(global_accs))

import pandas as pd

samples_all = chains.reshape(-1,n_dim)

labels = ['$z_0$', '$\sigma_z$', '$M_0$', '$\sigma_M$']

df = pd.DataFrame()
for i in range(n_param):
    df[labels[i]] = samples_all[:,i]

df = df.sample(n=10000)

import seaborn as sns

g = sns.pairplot(df, corner=True, kind='hist',
                 diag_kws=dict(common_norm=False, rasterized=True),
                 plot_kws=dict(common_norm=False))

for i in range(n_param):
    g.axes[i,i].axvline(true_param[i], color=sns.color_palette()[3])
    for j in range(i):
        g.axes[i,j].axvline(true_param[j], color=sns.color_palette()[3])
        g.axes[i,j].axhline(true_param[i], color=sns.color_palette()[3])

g.figure.savefig('./corner.pdf')
