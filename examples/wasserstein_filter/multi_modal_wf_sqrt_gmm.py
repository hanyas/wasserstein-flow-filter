import numpy as onp

import jax.numpy as jnp
import jax.random

import jaxopt
import matplotlib.pyplot as plt

from vwf.objects import MVNSqrt, GMMSqrt
from vwf.filters import wasserstein_filter_sqrt_gmm
from vwf.models.multi_modal_sqrt import build_model, generate_data

from vwf.sigma_points import monte_carlo_points
from vwf.utils import euler_odeint

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

s = jnp.array([1.0])

m0 = jnp.array([0.0])
P0_sqrt = jnp.eye(1)
x0 = MVNSqrt(m0, P0_sqrt)

T = 200

key = jax.random.PRNGKey(444)
key, sub_key = jax.random.split(key, 2)
true_states, observations = generate_data(sub_key, x0, T, s)

nb_comp = 2
mc_points = lambda key: monte_carlo_points(key, dim=1, nb_comp=nb_comp, nb_samples=500)

init_gmm = GMMSqrt(
    jnp.array([[-5.0], [5.0]]),
    jnp.repeat(P0_sqrt[None, :], nb_comp, axis=0),
)

trans_model, obsrv_model = build_model(s)

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
filt_states_mean = onp.array(filt_states.mean)
filt_states_cov_sqrt = onp.array(filt_states.cov_sqrt)
t = onp.arange(T + 1)

plt.figure()
plt.plot(t, true_states[:, 0], "k")
plt.plot(t, filt_states_mean[:, 0, 0], "red")
plt.plot(t, filt_states_mean[:, 1, 0], "blue")
plt.fill_between(
    t,
    filt_states_mean[:, 0, 0] - 2.0 * filt_states_cov_sqrt[:, 0, 0, 0],
    filt_states_mean[:, 0, 0] + 2.0 * filt_states_cov_sqrt[:, 0, 0, 0],
    color="tab:red",
    alpha=0.25,
)
plt.fill_between(
    t,
    filt_states_mean[:, 1, 0] - 2.0 * filt_states_cov_sqrt[:, 1, 0, 0],
    filt_states_mean[:, 1, 0] + 2.0 * filt_states_cov_sqrt[:, 1, 0, 0],
    color="tab:blue",
    alpha=0.25,
)
plt.show()
