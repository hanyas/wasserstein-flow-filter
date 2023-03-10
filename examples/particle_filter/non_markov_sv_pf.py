import numpy as onp

import jax.numpy as jnp
import jax.random

import matplotlib.pyplot as plt

from vwf.objects import MVNStandard
from vwf.filters import (
    non_markov_stratified_particle_filter as particle_filter,
)
from vwf.models.non_markov_sv import build_model, generate_data

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

mu = 0.5
a = 0.975
sig = jnp.sqrt(0.02)
rho = -0.8

m0 = jnp.array([mu])
P0 = jnp.diag(jnp.array([sig**2 / (1 - a**2)]))
x0 = MVNStandard(m0, P0)

T = 500
N = 500

key = jax.random.PRNGKey(123)
key, sub_key = jax.random.split(key, 2)

true_params = jnp.array([mu, a, sig, rho])
Xs, Ys = generate_data(sub_key, x0, T, true_params)

trns_mdl, obs_mdl = build_model(true_params)
Xf, ell, Ws = jax.jit(particle_filter, static_argnums=(1, 4, 5))(
    key, N, Ys, x0, trns_mdl, obs_mdl
)
print("Likelihood: ", ell)

#
Xs = onp.array(Xs)
Xf = onp.array(Xf)
t = onp.arange(T)

MEAN = onp.mean(Xf, axis=1)[:, 0]
VAR = (onp.average(Xf[..., 0] ** 2, axis=1, weights=Ws) - MEAN**2)
STD = VAR**0.5

plt.figure()
plt.plot(t, Xs, "k")
plt.plot(t, MEAN, "r")
plt.fill_between(
    t,
    MEAN - 2 * STD,
    MEAN + 2 * STD,
    color="tab:red",
    alpha=0.25,
)
plt.show()
