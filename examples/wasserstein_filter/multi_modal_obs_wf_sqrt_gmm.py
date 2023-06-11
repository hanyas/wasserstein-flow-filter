import numpy as onp

import jax.numpy as jnp
import jax.random

from wasserstein_filter.objects import MVNSqrt, GMMSqrt
from wasserstein_filter.filters import wasserstein_filter_sqrt_gmm
from wasserstein_filter.models.multi_modal_obs_sqrt import (
    build_model,
    generate_data,
)
from wasserstein_filter.numerics import gmm_monte_carlo_points
from wasserstein_filter.utils import euler_odeint

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


s = jnp.array([2.5])

m0 = jnp.array([0.0])
P0_sqrt = jnp.eye(1)
init_dist = MVNSqrt(m0, P0_sqrt)

nb_steps = 500

key = jax.random.PRNGKey(131)
key, sub_key = jax.random.split(key, 2)

trans_model, obsrv_model = build_model(s)
true_states, observations = generate_data(sub_key, init_dist, nb_steps, s)

nb_comp = 2
nb_samples = 500

mc_points = lambda key, mus, covs_sqrt: gmm_monte_carlo_points(
    key, mus, covs_sqrt, nb_comp, 1, nb_samples
)

init_gmm = GMMSqrt(
    jnp.array([[-5.0], [5.0]]),
    jnp.repeat(P0_sqrt[None, :], nb_comp, axis=0),
)


key, sub_key = jax.random.split(key, 2)
filt_states, ell = jax.jit(
    wasserstein_filter_sqrt_gmm, static_argnums=(3, 4, 5, 6, 7)
)(
    sub_key,
    observations,
    init_gmm,
    trans_model,
    obsrv_model,
    mc_points,
    euler_odeint,
    step_size=1e-2,
)
print("Likelihood: ", ell)

#
true_states = onp.array(true_states)
observations = onp.array(observations)
filt_states_mean = onp.array(filt_states.mean)
filt_states_cov_sqrt = onp.array(filt_states.cov_sqrt)
t = onp.arange(nb_steps + 1)

plt.figure()
plt.plot(t, true_states[:, 0], "k", linewidth=3.5)
plt.scatter(t[1:], observations[:, 0], c="tab:green", s=5.0)
plt.plot(t, filt_states_mean[:, 0, 0], "tab:blue")
plt.plot(t, filt_states_mean[:, 1, 0], "tab:red")
plt.fill_between(
    t,
    filt_states_mean[:, 0, 0] - 2.0 * filt_states_cov_sqrt[:, 0, 0, 0],
    filt_states_mean[:, 0, 0] + 2.0 * filt_states_cov_sqrt[:, 0, 0, 0],
    color="tab:blue",
    alpha=0.25,
)
plt.fill_between(
    t,
    filt_states_mean[:, 1, 0] - 2.0 * filt_states_cov_sqrt[:, 1, 0, 0],
    filt_states_mean[:, 1, 0] + 2.0 * filt_states_cov_sqrt[:, 1, 0, 0],
    color="tab:red",
    alpha=0.25,
)
plt.show()
