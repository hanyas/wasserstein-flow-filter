import jax.numpy as jnp
import jax.random

import matplotlib.pyplot as plt

from vwf.objects import MVNStandard
from vwf.filters import kalman_filter
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
Zf, ell = kalman_filter(Ys, z0, trns_mdl, obs_mdl)

print("Likelihood: ", ell)

plt.figure()
plt.plot(Zs[1:, 0], "k")
plt.plot(Zf.mean[1:, 0], "r")
plt.show()


def _tanh(x):
    return jnp.clip(jnp.tanh(x), -0.999, 0.999)


def _constrain(params):
    mu, a_aux, sig_aux, rho_aux = params
    a, rho = _tanh(a_aux), _tanh(rho_aux)
    sig = jnp.log1p(jnp.exp(sig_aux))
    return jnp.array([mu, a, sig, rho])


def log_likelihood(params, z0, Ys):
    trns_mdl, obs_mdl = build_model(_constrain(params))
    _, ell = kalman_filter(Ys, z0, trns_mdl, obs_mdl)
    return -ell


# solver = jaxopt.ScipyMinimize(fun=log_likelihood, tol=1e-4, jit=True)
# init_params = jnp.array([0.0, 0.0, 0.0, 0.0])
# res = solver.run(init_params, z0=z0, Ys=Ys)
# print(_transform(res.params))
