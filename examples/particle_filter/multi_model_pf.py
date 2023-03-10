import jax.numpy as jnp
import jax.random

import matplotlib.pyplot as plt

from vwf.objects import MVNStandard
from vwf.filters import particle_filter
from vwf.models.multi_modal import build_model, generate_data

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

m0 = jnp.array([0.0])
P0 = jnp.eye(1)
x0 = MVNStandard(m0, P0)

T = 500
N = 10_000

key = jax.random.PRNGKey(123)
key, sub_key = jax.random.split(key, 2)

true_params = jnp.array([1.0])
Xs, Ys = generate_data(sub_key, x0, T, true_params)

trns_mdl, obs_mdl = build_model(true_params)
Xf, ell, Ws = jax.jit(particle_filter, static_argnums=(1, 4, 5))(
    key, N, Ys, x0, trns_mdl, obs_mdl
)
print("Likelihood: ", ell)

t = jnp.repeat(jnp.arange(T + 1)[:, None], N, axis=1)

plt.figure()
plt.plot(t, Xs, "k")
plt.scatter(t, Xf, alpha=0.01, c="tab:blue")
plt.show()
