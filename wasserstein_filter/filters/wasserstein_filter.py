from typing import Callable

import jax
import jax.random
from jax.flatten_util import ravel_pytree

import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn

from wasserstein_filter.objects import MVNStandard, ConditionalMVN
from wasserstein_filter.utils import fixed_point, rk4_odeint, euler_odeint
from wasserstein_filter.utils import none_or_concat


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


def log_posterior(
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
    ode_integrator: Callable,
    step_size: float,
):
    d = dist.dim
    gradP = jax.grad(log_posterior)

    _dist, _unflatten = ravel_pytree(dist)

    def _ode(t, x):
        mu, sigma = _unflatten(x)

        z, w = sigma_points(mu, jnp.linalg.cholesky(sigma))
        dP = jax.vmap(gradP, in_axes=(0, None, None, None))(
            z, observation, prior, observation_model
        )

        mu_dt = jnp.einsum("nk,n->k", dP, w)
        sigma_dt = (
            2.0 * jnp.eye(d)
            + jnp.einsum("nk,nh,n->kh", dP, z - mu, w)
            + jnp.einsum("nk,nh,n->kh", z - mu, dP, w)
        )

        dx_dt = MVNStandard(mu_dt, sigma_dt)
        return ravel_pytree(dx_dt)[0]

    _dist = ode_integrator(func=_ode, tk=0.0, yk=_dist, dt=step_size)
    return _unflatten(_dist)


def integrate_ode(
    prior: MVNStandard,
    observation: jnp.ndarray,
    observation_model: ConditionalMVN,
    sigma_points: Callable,
    ode_integrator: Callable,
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
            ode_integrator,
            step_size,
        )

    return fixed_point(fun_to_iter, prior, criterion)


def wasserstein_filter(
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: ConditionalMVN,
    observation_model: ConditionalMVN,
    sigma_points: Callable,
    ode_integrator: Callable = euler_odeint,
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
            ode_integrator,
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
