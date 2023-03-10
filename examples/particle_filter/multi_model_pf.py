import numpy as onp

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
N = 500

key = jax.random.PRNGKey(123)
key, sub_key = jax.random.split(key, 2)

s = jnp.array([1.0])
Xs, Ys = generate_data(sub_key, x0, T, s)

trns_mdl, obs_mdl = build_model(s)
Xf, ell, Ws = jax.jit(particle_filter, static_argnums=(1, 4, 5))(
    key, N, Ys, x0, trns_mdl, obs_mdl
)
print("Likelihood: ", ell)

#
Xs = onp.array(Xs)
Xf = onp.array(Xf)

t = onp.arange(T + 1)
xf_pos = onp.array([x[x >= 0.].mean() for x in Xf])
xf_neg = onp.array([x[x < 0.].mean() for x in Xf])

plt.figure()
plt.plot(t, Xs, "k")
plt.plot(t, xf_pos, alpha=1., c="tab:blue")
plt.plot(t, xf_neg, alpha=1., c="tab:red")
plt.show()
