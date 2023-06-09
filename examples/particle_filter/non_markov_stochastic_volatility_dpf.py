import os

import numpy as onp

import jax
import jax.numpy as jnp
import jax.random

import jaxopt

from wasserstein_filter.objects import MVNStandard
from wasserstein_filter.filters import (
    non_markov_diffable_particle_filter as particle_filter,
)
from wasserstein_filter.models.non_markov_stochastic_volatility import (
    build_model,
    generate_data,
)

import dask
from dask import delayed
from dask.diagnostics import ProgressBar

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
)

mu = 0.5
a = 0.975
sig = jnp.sqrt(0.02)
rho = -0.8

true_params = jnp.array([mu, a, sig, rho])

m0 = jnp.array([mu])
P0 = jnp.diag(jnp.array([sig**2 / (1 - a**2)]))
init_dist = MVNStandard(m0, P0)

nb_steps = 500

key = jax.random.PRNGKey(123)
key, sub_key = jax.random.split(key, 2)
true_states, observations = generate_data(
    sub_key, init_dist, nb_steps, true_params
)

nb_particles = 500
trans_mdl, obsrv_mdl = build_model(true_params)
filt_states, ell, weights = jax.jit(particle_filter, static_argnums=(1, 4, 5))(
    key, nb_particles, observations, init_dist, trans_mdl, obsrv_mdl
)
print("Likelihood: ", ell)

#
true_states = onp.array(true_states)
filt_states = onp.array(filt_states)
weights = onp.array(weights)
t = onp.arange(nb_steps)

MEAN = onp.average(filt_states[..., 0], axis=-1, weights=weights)
VAR = onp.average(filt_states[..., 0]**2, axis=-1, weights=weights) - MEAN**2
STD = VAR**0.5

plt.figure()
plt.plot(t, true_states[:, 0], "k")
plt.plot(t, MEAN, "r")
plt.fill_between(
    t,
    MEAN - 2 * STD,
    MEAN + 2 * STD,
    color="tab:red",
    alpha=0.25,
)
plt.show()


def _tanh(x):
    return jnp.clip(jnp.tanh(x), -0.999, 0.999)


def _constrain(params):
    mu, a_aux, sig_aux, rho_aux = params
    a, rho = _tanh(a_aux), _tanh(rho_aux)
    sig = jnp.log1p(jnp.exp(sig_aux))
    return jnp.array([mu, a, sig, rho])


def log_likelihood(params, observations, init_dist, seed):
    trans_mdl, obsrv_mdl = build_model(_constrain(params))
    key = jax.random.PRNGKey(seed)
    _, ell, _ = particle_filter(
        key, nb_particles, observations, init_dist, trans_mdl, obsrv_mdl
    )
    return -ell


def optimization_loop(seed):
    solver = jaxopt.ScipyMinimize(fun=log_likelihood, tol=1e-4, jit=True)
    init_params = jnp.array([0.0, 0.0, 0.0, 0.0])
    res = solver.run(
        init_params, observations=observations, init_dist=init_dist, seed=seed
    )
    return res.params


N_OPT = 25
results = [delayed(optimization_loop)(1989 + i) for i in range(N_OPT)]
with ProgressBar():
    params_hst = dask.compute(*results)
params_hst = jax.vmap(_constrain)(jnp.stack(params_hst))

results_df = pd.DataFrame(
    data=params_hst, columns=[r"$\mu$", r"$\alpha$", r"$\sigma$", r"$\rho$"]
)

g = sns.PairGrid(results_df)
g.map_upper(sns.kdeplot, fill=True)
g.map_lower(sns.kdeplot, fill=True)
g.map_diag(sns.kdeplot)
plt.show()
