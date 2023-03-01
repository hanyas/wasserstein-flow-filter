import jax.numpy as jnp
import jax.random

import matplotlib.pyplot as plt

from vwf.objects import MVNStandard
from vwf.filters import wasserstein_filter
from vwf.models.markov_sv import build_model, generate_data

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


mu = 0.5
a = 0.975
sig = jnp.sqrt(0.02)
rho = -0.8

m0 = jnp.array([mu, 0.0])
P0 = jnp.diag(jnp.array([sig**2 / (1 - a**2), 1.0]))
z0 = MVNStandard(m0, P0)

T = 500

key = jax.random.PRNGKey(123)
key, sub_key = jax.random.split(key, 2)

true_params = jnp.array([mu, a, sig, rho])
Zs, Ys = generate_data(sub_key, z0, T, true_params)

N = 250
dt = 1e-2

key, sub_key = jax.random.split(key, 2)
rv = jax.random.normal(sub_key, shape=(T, N, 2))

trns_mdl, obs_mdl = build_model(true_params)
Zf, ell = jax.jit(wasserstein_filter,
                  static_argnums=(3, 4, 5, 6))(Ys, rv, z0,
                                               trns_mdl, obs_mdl,
                                               dt, lambda i, *_: i < 500)
print("Likelihood: ", ell)

plt.figure()
plt.plot(Zs[1:, 0], 'k')
plt.plot(Zf.mean[1:, 0], 'r')
plt.show()
