import numpy as onp

import jax.numpy as jnp
import jax.random

import jaxopt
import matplotlib.pyplot as plt

from vwf.objects import MVNSqrt
from vwf.filters import wasserstein_filter_sqrt
from vwf.models.markov_sv_sqrt import build_model, generate_data

from vwf.sigma_points import cubature_points
from vwf.sigma_points import gauss_hermite_points
from vwf.utils import euler_odeint

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

mu = 0.5
a = 0.975
sig = jnp.sqrt(0.02)
rho = -0.8

true_params = jnp.array([mu, a, sig, rho])

m0 = jnp.array([mu, 0.0])
P0_sqrt = jnp.diag(jnp.array([sig / jnp.sqrt(1 - a**2), 1.0]))
init_dist = MVNSqrt(m0, P0_sqrt)

nb_steps = 500

key = jax.random.PRNGKey(123)
key, sub_key = jax.random.split(key, 2)
true_states, observations = generate_data(sub_key, init_dist, nb_steps, true_params)

# sigma_points = lambda mu, cov_sqrt: cubature_points(mu, cov_sqrt)
sigma_points = lambda mu, cov_sqrt: gauss_hermite_points(mu, cov_sqrt, order=5)

trans_model, obsrv_model = build_model(true_params)
filt_states, ell = jax.jit(
    wasserstein_filter_sqrt, static_argnums=(2, 3, 4, 5)
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

#
true_state = onp.array(true_states)
filt_states_mean = onp.array(filt_states.mean)
filt_states_cov_sqrt = onp.array(filt_states.cov_sqrt)
t = onp.arange(nb_steps + 1)

plt.figure()
plt.plot(t, true_states[:, 0], "k")
plt.plot(t, filt_states_mean[:, 0], "r")
plt.fill_between(
    t,
    filt_states_mean[:, 0] - 2. * filt_states_cov_sqrt[:, 0, 0],
    filt_states_mean[:, 0] + 2. * filt_states_cov_sqrt[:, 0, 0],
    color="tab:red",
    alpha=0.25
)
plt.show()


# def _tanh(x):
#     return jnp.clip(jnp.tanh(x), -0.999, 0.999)
#
#
# def _constrain(params):
#     mu, a_aux, sig_aux, rho_aux = params
#     a, rho = _tanh(a_aux), _tanh(rho_aux)
#     sig = jnp.log1p(jnp.exp(sig_aux))
#     return jnp.array([mu, a, sig, rho])
#
#
# def log_likelihood(params, init_dist, observations):
#     trans_model, obsrv_model = build_model(_constrain(params))
#     _, ell = wasserstein_filter_sqrt(
#         observations,
#         init_dist,
#         trans_model,
#         obsrv_model,
#         sigma_points,
#         euler_odeint,
#         step_size=1e-2,
#     )
#     return -ell
#
#
# solver = jaxopt.ScipyMinimize(
#     fun=log_likelihood,
#     method="SLSQP",
#     options={"disp": True},
#     jit=True,
#     maxiter=15,
# )
#
# init_params = jnp.array([0.0, 0.0, 0.0, 0.0])
# opt_params = solver.run(init_params, init_dist, observations).params
# print(_constrain(opt_params))
