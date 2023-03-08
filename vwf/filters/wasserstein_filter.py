from typing import Callable

import jax
import jax.random
from jax.flatten_util import ravel_pytree

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal as mvn

from vwf.objects import MVNStandard, ConditionalModel
from vwf.utils import fixed_point, rk4_odeint, euler_odeint
from vwf.utils import kullback_leibler_cond, wasserstein_cond


def linearize(model: ConditionalModel, x: MVNStandard):
    mean, cov = model
    m, p = x

    F = jax.jacfwd(mean, 0)(m)
    b = mean(m) - F @ m
    Q = cov(m)
    return F, b, Q


def predict(F, b, Q, x):
    m, P = x
    m = F @ m + b
    P = Q + F @ P @ F.T
    return MVNStandard(m, P)


def potential(x, y, prior, obs_mdl):
    m, P = prior
    mean_fcn, cov_fcn = obs_mdl
    mean_y, cov_y = mean_fcn(x), cov_fcn(x)
    return -mvn.logpdf(y, mean_y, cov_y) - mvn.logpdf(x, m, P)


def ode(
    dist: MVNStandard,
    prior: MVNStandard,
    observation: jnp.ndarray,
    observation_model: ConditionalModel,
    sigma_points: Callable,
    integrator: Callable,
    step_size: float,
):
    d = dist.dim
    gradV = jax.grad(potential)

    _dist, _unflatten = ravel_pytree(dist)

    def _ode_fcn(t, x):
        mu, sigma = _unflatten(x)

        z, w = sigma_points(mu, jnp.linalg.cholesky(sigma))

        dV = jax.vmap(gradV, in_axes=(0, None, None, None))(
            z, observation, prior, observation_model
        )

        mu_dt = -jnp.einsum("nk,n->k", dV, w)
        sigma_dt = (
            2.0 * jnp.eye(d)
            - jnp.einsum("nk,nh,n->kh", dV, z - mu, w)
            - jnp.einsum("nk,nh,n->kh", z - mu, dV, w)
        )

        dx_dt = MVNStandard(mu_dt, sigma_dt)
        return ravel_pytree(dx_dt)[0]

    _dist = integrator(_ode_fcn, 0.0, _dist, dt=step_size)
    return _unflatten(_dist)


def integrate_ode(
    prior: MVNStandard,
    observation: jnp.ndarray,
    observation_model: ConditionalModel,
    sigma_points: Callable,
    integrator: Callable,
    step_size: float,
    criterion: Callable,
):
    def fun_to_iter(dist):
        return ode(
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
    transition_model: ConditionalModel,
    observation_model: ConditionalModel,
    sigma_points: Callable,
    integrator: Callable = euler_odeint,
    step_size: float = 1e-2,
    stopping_criterion: Callable = kullback_leibler_cond,
):
    def _cond_log_pdf(x, y, obs_mdl):
        mean_fcn, cov_fcn = obs_mdl
        mean_y, cov_y = mean_fcn(x), cov_fcn(x)
        return mvn.logpdf(y, mean_y, cov_y)

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
        log_pdfs = jax.vmap(_cond_log_pdf, in_axes=(0, None, None))(
            z, y, observation_model
        )
        ell += jnp.log(jnp.average(jnp.exp(log_pdfs), weights=w))

        return (xf, ell), xf

    x0 = initial_dist
    ys = observations

    (_, ell), xf = jax.lax.scan(body, (x0, 0.0), xs=ys)
    return xf, ell
