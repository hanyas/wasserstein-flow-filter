import jax.numpy as jnp
import jax.random

import matplotlib.pyplot as plt

from vwf.objects import MVNStandard
from vwf.filters import particle_filter
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

trns_mdl, obs_mdl = build_model(true_params)
Zf, ell, Ws = jax.jit(particle_filter, static_argnums=(1, 4, 5))(
    key, 250, Ys, z0, trns_mdl, obs_mdl
)
print("Likelihood: ", ell)

MEAN_PARTICLES = jnp.mean(Zf, axis=1)[1:, 0]
VAR_PARTICLES = (
    jnp.average(Zf[1:, :, 0] ** 2, axis=1, weights=Ws) - MEAN_PARTICLES**2
)
STD_PARTICLES = VAR_PARTICLES**0.5

plt.figure()
plt.plot(Zs[1:, 0], "k")
plt.plot(MEAN_PARTICLES, "r")
# plt.fill_between(jnp.arange(T), MEAN_PARTICLES
#                  - 2 * STD_PARTICLES, MEAN_PARTICLES + 2 * STD_PARTICLES,
#                  color="tab:blue", alpha=0.5)
plt.show()
