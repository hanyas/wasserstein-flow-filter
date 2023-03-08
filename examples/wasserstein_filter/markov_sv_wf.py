import jax.numpy as jnp
import jax.random

import matplotlib.pyplot as plt

from vwf.objects import MVNStandard
from vwf.filters import wasserstein_filter
from vwf.models.markov_sv import build_model, generate_data

from vwf.sigma_points import cubature_points
from vwf.sigma_points import gauss_hermite_points
from vwf.sigma_points import monte_carlo_points

from vwf.utils import euler_odeint

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


mu = 0.5
a = 0.975
sig = jnp.sqrt(0.02)
rho = -0.8

m0 = jnp.array([mu, 0.0])
P0 = jnp.diag(jnp.array([sig**2 / (1 - a**2), 1.0]))
init_dist = MVNStandard(m0, P0)

T = 500

key = jax.random.PRNGKey(123)
key, sub_key = jax.random.split(key, 2)

true_params = jnp.array([mu, a, sig, rho])
true_states, observations = generate_data(sub_key, init_dist, 500, true_params)

# key, sub_key = jax.random.split(key, 2)
# rv = jax.random.normal(sub_key, shape=(64, 2))
# sigma_points = lambda mu, cov_sqrt: monte_carlo_points(mu, cov_sqrt, rv)

# sigma_points = lambda mu, cov_sqrt: cubature_points(mu, cov_sqrt)
sigma_points = lambda mu, cov_sqrt: gauss_hermite_points(mu, cov_sqrt, order=5)


trans_model, obsrv_model = build_model(true_params)
filt_states, ell = jax.jit(
    wasserstein_filter, static_argnums=(2, 3, 4, 5)
)(
    observations,
    init_dist,
    trans_model,
    obsrv_model,
    sigma_points,
    euler_odeint,
    step_size=1e-2,
)
print("Likelihood: ", ell)

plt.figure()
plt.plot(true_states[1:, 0], "k")
plt.plot(filt_states.mean[1:, 0], "r")
plt.show()
