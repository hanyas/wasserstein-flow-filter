import jax.numpy as jnp
import jax.random

import jaxopt
import matplotlib.pyplot as plt

from vwf.objects import MVNSqrt
from vwf.filters import wasserstein_filter_sqrt
from vwf.models.markov_sv_sqrt import build_model, generate_data

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

mu = 0.5
a = 0.975
sig = jnp.sqrt(0.02)
rho = -0.8

m0 = jnp.array([mu, 0.0])
P0_sqrt = jnp.diag(jnp.array([sig / jnp.sqrt(1 - a**2), 1.0]))
z0 = MVNSqrt(m0, P0_sqrt)

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
Zf, ell = jax.jit(wasserstein_filter_sqrt,
                  static_argnums=(3, 4, 5, 6))(Ys, rv, z0,
                                               trns_mdl, obs_mdl,
                                               dt, lambda i, *_: i < 500)
print("Likelihood: ", ell)

plt.figure()
plt.plot(Zs[1:, 0], 'k')
plt.plot(Zf.mean[1:, 0], 'r')
plt.show()


def _tanh(x):
    return jnp.clip(jnp.tanh(x), -0.999, 0.999)


def _constrain(params):
    mu, a_aux, sig_aux, rho_aux = params
    a, rho = _tanh(a_aux), _tanh(rho_aux)
    sig = jnp.log1p(jnp.exp(sig_aux))
    return jnp.array([mu, a, sig, rho])


def log_likelihood(params, z0, Ys, rv):
    trns_mdl, obs_mdl = build_model(_constrain(params))
    _, ell = wasserstein_filter_sqrt(Ys, rv, z0, trns_mdl, obs_mdl, dt)
    return - ell


solver = jaxopt.ScipyMinimize(fun=log_likelihood,
                              method='SLSQP',
                              options={'disp': True},
                              jit=True, maxiter=15)

init_params = jnp.array([0.0, 0.0, 0.0, 0.0])
res = solver.run(init_params, z0=z0, Ys=Ys, rv=rv)
print(_constrain(res.params))
