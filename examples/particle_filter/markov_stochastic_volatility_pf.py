import numpy as onp

import jax.numpy as jnp
import jax.random

from wasserstein_filter.objects import MVNStandard
from wasserstein_filter.filters import particle_filter
from wasserstein_filter.models.markov_stochastic_volatility import (
    build_model,
    generate_data,
)

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


mu = 0.5
a = 0.975
sig = jnp.sqrt(0.02)
rho = -0.8

true_params = jnp.array([mu, a, sig, rho])

m0 = jnp.array([mu, 0.0])
P0 = jnp.diag(jnp.array([sig**2 / (1 - a**2), 1.0]))
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
t = onp.arange(nb_steps + 1)

MEAN = onp.average(filt_states[..., 0], axis=-1, weights=weights)
VAR = onp.average(filt_states[..., 0] ** 2, axis=-1, weights=weights) - MEAN**2
STD = VAR**0.5

plt.figure()
plt.plot(t, true_states[..., 0], "k")
plt.plot(t, MEAN, "r")
plt.fill_between(
    t,
    MEAN - 2 * STD,
    MEAN + 2 * STD,
    color="tab:red",
    alpha=0.25,
)
plt.show()
