import numpy as onp

import jax.numpy as jnp
import jax.random

from wasserstein_filter.objects import MVNStandard
from wasserstein_filter.filters import particle_filter
from wasserstein_filter.models.multi_modal_obs import build_model, generate_data

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


s = jnp.array([2.5])

m0 = jnp.array([0.0])
P0 = jnp.eye(1)
init_dist = MVNStandard(m0, P0)

nb_steps = 500

key = jax.random.PRNGKey(131)
key, sub_key = jax.random.split(key, 2)
true_states, observations = generate_data(sub_key, init_dist, nb_steps, s)

nb_particles = 500
trans_mdl, obsrv_mdl = build_model(s)
filt_states, ell, weights = jax.jit(particle_filter, static_argnums=(1, 4, 5))(
    key, nb_particles, observations, init_dist, trans_mdl, obsrv_mdl
)
print("Likelihood: ", ell)

#
true_states = onp.array(true_states)
observations = onp.array(observations)
filt_states = onp.array(filt_states)

t = onp.arange(nb_steps + 1)
xf_pos_mean = onp.array([x[x >= 0.].mean() for x in filt_states])
xf_neg_mean = onp.array([x[x < 0.].mean() for x in filt_states])

xf_pos_std = onp.array([x[x >= 0.].std() for x in filt_states])
xf_neg_std = onp.array([x[x < 0.].std() for x in filt_states])

plt.figure()
plt.plot(t, true_states, "k", linewidth=3.5)
plt.scatter(t[1:], observations[:, 0], c="tab:green", s=5.0)
plt.plot(t, xf_pos_mean, "tab:red")
plt.plot(t, xf_neg_mean, "tab:blue")
plt.fill_between(
    t,
    xf_pos_mean - 2.0 * xf_pos_std,
    xf_pos_mean + 2.0 * xf_pos_std,
    color="tab:red",
    alpha=0.25,
)
plt.fill_between(
    t,
    xf_neg_mean - 2.0 * xf_neg_std,
    xf_neg_mean + 2.0 * xf_neg_std,
    color="tab:blue",
    alpha=0.25,
)
plt.show()
