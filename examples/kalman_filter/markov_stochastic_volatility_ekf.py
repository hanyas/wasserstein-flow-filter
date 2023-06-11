import numpy as onp

import jax.numpy as jnp
import jax.random

import jaxopt

from wasserstein_filter.objects import MVNStandard
from wasserstein_filter.filters import kalman_filter
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

trans_mdl, obsrv_mdl = build_model(true_params)
true_states, observations = generate_data(
    sub_key, init_dist, nb_steps, true_params
)

filt_states, ell = kalman_filter(observations, init_dist, trans_mdl, obsrv_mdl)

print("Likelihood: ", ell)

#
true_states = onp.array(true_states)
filt_states_mean = onp.array(filt_states.mean)

plt.figure()
plt.plot(true_states[1:, 0], "k")
plt.plot(filt_states_mean[1:, 0], "r")
plt.show()


def _tanh(x):
    return jnp.clip(jnp.tanh(x), -0.999, 0.999)


def _constrain(params):
    mu, a_aux, sig_aux, rho_aux = params
    a, rho = _tanh(a_aux), _tanh(rho_aux)
    sig = jnp.log1p(jnp.exp(sig_aux))
    return jnp.array([mu, a, sig, rho])


def log_likelihood(params, observations, init_dist):
    trans_mdl, obsrv_mdl = build_model(_constrain(params))
    _, ell = kalman_filter(observations, init_dist, trans_mdl, obsrv_mdl)
    return -ell


solver = jaxopt.ScipyMinimize(fun=log_likelihood, tol=1e-4, jit=True)
init_params = jnp.array([0.0, 0.0, 0.0, 0.0])
res = solver.run(init_params, observations=observations, init_dist=init_dist)
print("Parameter estimate: ", _constrain(res.params))
