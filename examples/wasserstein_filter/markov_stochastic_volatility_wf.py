import numpy as onp

import jax.numpy as jnp
import jax.random

from wasserstein_filter.objects import MVNStandard
from wasserstein_filter.filters import wasserstein_filter
from wasserstein_filter.models.markov_stochastic_volatility import (
    build_model,
    generate_data,
)
from wasserstein_filter.numerics import cubature_points
from wasserstein_filter.numerics import gauss_hermite_points
from wasserstein_filter.utils import euler_odeint

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


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

trans_model, obsrv_model = build_model(true_params)
true_states, observations = generate_data(
    sub_key, init_dist, nb_steps, true_params
)

# sigma_points = lambda mu, cov_sqrt: cubature_points(mu, cov_sqrt)
sigma_points = lambda mu, cov_sqrt: gauss_hermite_points(mu, cov_sqrt, order=5)

filt_states, ell = jax.jit(wasserstein_filter, static_argnums=(2, 3, 4, 5, 6))(
    observations,
    init_dist,
    trans_model,
    obsrv_model,
    sigma_points,
    euler_odeint,
    step_size=1e-2,
)
print("Likelihood: ", ell)

#
true_state = onp.array(true_states)
filt_states_mean = onp.array(filt_states.mean)
filt_states_cov = onp.array(filt_states.cov)
t = onp.arange(nb_steps + 1)

plt.figure()
plt.plot(t, true_state[:, 0], "k")
plt.plot(t, filt_states_mean[:, 0], "r")
plt.fill_between(
    t,
    filt_states_mean[:, 0] - 2.0 * filt_states_cov[:, 0, 0] ** 0.5,
    filt_states_mean[:, 0] + 2.0 * filt_states_cov[:, 0, 0] ** 0.5,
    color="tab:red",
    alpha=0.25,
)
plt.show()
