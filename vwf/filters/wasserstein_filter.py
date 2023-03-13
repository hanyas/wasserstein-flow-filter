from typing import Callable

import jax
import jax.random
from jax.flatten_util import ravel_pytree

import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn
from jax.experimental.ode import odeint

from vwf.objects import MVNStandard, ConditionalMVN
from vwf.utils import fixed_point, rk4_odeint, euler_odeint
from vwf.utils import kullback_leibler_mvn_cond, wasserstein_mvn_cond
from vwf.utils import none_or_concat


def linearize(model: ConditionalMVN, x: MVNStandard):
    mean_fcn, cov_fcn = model
    m, p = x

    F = jax.jacfwd(mean_fcn, 0)(m)
    b = mean_fcn(m) - F @ m
    Q = cov_fcn(m)
    return F, b, Q


def predict(F, b, Q, x):
    m, P = x
    m = F @ m + b
    P = Q + F @ P @ F.T
    return MVNStandard(m, P)


def log_target(
    # Log-target is the negative potential V
    state: jnp.ndarray,
    observation: jnp.ndarray,
    prior: MVNStandard,
    observation_model: ConditionalMVN,
):
    m, P = prior
    return (
        observation_model.logpdf(state, observation)
        + mvn.logpdf(state, m, P)
    )


def ode_step(
    dist: MVNStandard,
    prior: MVNStandard,
    observation: jnp.ndarray,
    observation_model: ConditionalMVN,
    sigma_points: Callable,
    integrator: Callable,
    step_size: float,
):
    d = dist.dim
    gradV = jax.grad(log_target)

    _dist, _unflatten = ravel_pytree(dist)

    def _ode(t, x):
        mu, sigma = _unflatten(x)

        z, w = sigma_points(mu, jnp.linalg.cholesky(sigma))

        dV = jax.vmap(gradV, in_axes=(0, None, None, None))(
            z, observation, prior, observation_model
        )

        mu_dt = jnp.einsum("nk,n->k", dV, w)
        sigma_dt = (
            2.0 * jnp.eye(d)
            + jnp.einsum("nk,nh,n->kh", dV, z - mu, w)
            + jnp.einsum("nk,nh,n->kh", z - mu, dV, w)
        )

        dx_dt = MVNStandard(mu_dt, sigma_dt)
        return ravel_pytree(dx_dt)[0]

    _dist = integrator(func=_ode, tk=0.0, yk=_dist, dt=step_size)
    return _unflatten(_dist)


def integrate_ode(
    prior: MVNStandard,
    observation: jnp.ndarray,
    observation_model: ConditionalMVN,
    sigma_points: Callable,
    integrator: Callable,
    step_size: float,
    criterion: Callable,
):
    def fun_to_iter(dist):
        return ode_step(
            dist,
            prior,
            observation,
            observation_model,
            sigma_points,
            integrator,
            step_size,
        )

    return fixed_point(fun_to_iter, prior, criterion)


def wasserstein_filter(
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: ConditionalMVN,
    observation_model: ConditionalMVN,
    sigma_points: Callable,
    integrator: Callable = euler_odeint,
    step_size: float = 1e-2,
    stopping_criterion: Callable = lambda i, *_: i < 500,
):
    def body(carry, args):
        x, ell = carry
        y = args

        # predict
        F, b, Q = linearize(transition_model, x)
        xp = predict(F, b, Q, x)

        # innovate
        xf = integrate_ode(
            xp,
            y,
            observation_model,
            sigma_points,
            integrator,
            step_size,
            stopping_criterion,
        )

        # ell
        mu, sigma = xp
        z, w = sigma_points(mu, jnp.linalg.cholesky(sigma))
        log_pdfs = jax.vmap(observation_model.logpdf, in_axes=(0, None))(z, y)
        ell += jnp.log(jnp.average(jnp.exp(log_pdfs), weights=w))

        return (xf, ell), xf

    x0 = initial_dist
    ys = observations

    (_, ell), Xf = jax.lax.scan(body, (x0, 0.0), xs=ys)
    Xf = none_or_concat(Xf, x0, 1)
    return Xf, ell
