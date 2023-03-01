import os

import jax
import jax.numpy as jnp
import jax.random

import jaxopt

import dask
from dask import delayed
from dask.diagnostics import ProgressBar

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from vwf.objects import MVNStandard
from vwf.filters import non_markov_diffable_particle_filter as particle_filter
from vwf.models.non_markov_sv import build_model, generate_data

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

mu = 0.5
a = 0.975
sig = jnp.sqrt(0.02)
rho = -0.8

m0 = jnp.array([mu])
P0 = jnp.diag(jnp.array([sig**2 / (1 - a**2)]))
x0 = MVNStandard(m0, P0)

T = 500

key = jax.random.PRNGKey(123)
key, sub_key = jax.random.split(key, 2)

true_params = jnp.array([mu, a, sig, rho])
Xs, Ys = generate_data(sub_key, x0, T, true_params)

trns_mdl, obs_mdl = build_model(true_params)
Xf, ell, Ws = jax.jit(particle_filter,
                      static_argnums=(1, 4, 5))(key, 250, Ys, x0,
                                                trns_mdl, obs_mdl)
print("Likelihood: ", ell)

MEAN_PARTICLES = jnp.mean(Xf, axis=1)[:, 0]
VAR_PARTICLES = jnp.average(Xf[..., 0] ** 2, axis=1, weights=Ws) - MEAN_PARTICLES ** 2
STD_PARTICLES = VAR_PARTICLES ** 0.5

plt.figure()
plt.plot(Xs, 'k')
plt.plot(MEAN_PARTICLES, 'r')
# plt.fill_between(jnp.arange(T), MEAN_PARTICLES
#                  - 2 * STD_PARTICLES, MEAN_PARTICLES + 2 * STD_PARTICLES,
#                  color="tab:blue", alpha=0.5)
plt.show()


def _tanh(x):
    return jnp.clip(jnp.tanh(x), -0.999, 0.999)


def _constrain(params):
    mu, a_aux, sig_aux, rho_aux = params
    a, rho = _tanh(a_aux), _tanh(rho_aux)
    sig = jnp.log1p(jnp.exp(sig_aux))
    return jnp.array([mu, a, sig, rho])


def log_likelihood(params, x0, Ys, seed):
    trns_mdl, obs_mdl = build_model(_constrain(params))
    key = jax.random.PRNGKey(seed)
    _, ell, _ = particle_filter(key, 500, Ys, x0,
                                trns_mdl, obs_mdl)
    return - ell


def optimization_loop(seed):
    solver = jaxopt.ScipyMinimize(fun=log_likelihood,
                                  tol=1e-4, jit=True)

    init_params = jnp.array([0.0, 0.0, 0.0, 0.0])
    res = solver.run(init_params, x0=x0, Ys=Ys, seed=seed)
    return res.params


N_OPT = 25
results = [delayed(optimization_loop)(1989 + i) for i in range(N_OPT)]
with ProgressBar():
    params_hst = dask.compute(*results)
params_hst = jax.vmap(_constrain)(jnp.stack(params_hst))

results_df = pd.DataFrame(data=params_hst,
                          columns=[r"$\mu$", r"$\alpha$",
                                   r"$\sigma$", r"$\rho$"])

g = sns.PairGrid(results_df)
g.map_upper(sns.kdeplot, fill=True)
g.map_lower(sns.kdeplot, fill=True)
g.map_diag(sns.kdeplot)
plt.show()
