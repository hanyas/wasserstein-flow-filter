from typing import Callable

import jax
import jax.random
from jax.flatten_util import ravel_pytree

import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn
from jax.experimental.ode import odeint

from wasserstein_filter.objects import MVNSqrt, ConditionalMVNSqrt
from wasserstein_filter.utils import fixed_point, rk4_odeint, euler_odeint
from wasserstein_filter.utils import kullback_leibler_mvn_sqrt_cond, wasserstein_mvn_sqrt_cond
from wasserstein_filter.utils import tria_qr, tria_tril
from wasserstein_filter.utils import none_or_concat


def linearize(model: ConditionalMVNSqrt, x: MVNSqrt):
    mean_fcn, cov_sqrt_fcn = model
    m, _ = x

    F = jax.jacfwd(mean_fcn, 0)(m)
    b = mean_fcn(m) - F @ m
    Q_sqrt = cov_sqrt_fcn(m)
    return F, b, Q_sqrt


def predict(F, b, Q_sqrt, x):
    m, P_sqrt = x
    m = F @ m + b
    P_sqrt = tria_qr(jnp.concatenate([F @ P_sqrt, Q_sqrt], axis=1))
    return MVNSqrt(m, P_sqrt)


def log_target(
    # Log-target is the negative potential V
    state: jnp.ndarray,
    observation: jnp.ndarray,
    prior_sqrt: MVNSqrt,
    observation_model: ConditionalMVNSqrt,
):
    m, P_sqrt = prior_sqrt
    return (
            observation_model.logpdf(state, observation)
            + mvn.logpdf(state, m, P_sqrt @ P_sqrt.T)
    )


def ode_step(
    dist_sqrt: MVNSqrt,
    prior_sqrt: MVNSqrt,
    observation: jnp.ndarray,
    observation_model: ConditionalMVNSqrt,
    sigma_points: Callable,
    integrator: Callable,
    step_size: float,
):
    d = dist_sqrt.dim
    gradV = jax.grad(log_target)

    _dist_sqrt, _unflatten = ravel_pytree(dist_sqrt)

    def _ode(t, x):
        mu, sigma_sqrt = _unflatten(x)

        z, w = sigma_points(mu, sigma_sqrt)

        dV = jax.vmap(gradV, in_axes=(0, None, None, None))(
            z, observation, prior_sqrt, observation_model
        )

        sigma_dt = (
            2.0 * jnp.eye(d)
            + jnp.einsum("nk,nh,n->kh", dV, z - mu, w)
            + jnp.einsum("nk,nh,n->kh", z - mu, dV, w)
        )

        sigma_sqrt_inv = jnp.linalg.inv(sigma_sqrt)

        mu_dt = jnp.einsum("nk,n->k", dV, w)
        sigma_sqrt_dt = sigma_sqrt @ tria_tril(
            sigma_sqrt_inv @ sigma_dt @ sigma_sqrt_inv.T
        )

        dx_dt = MVNSqrt(mu_dt, sigma_sqrt_dt)
        return ravel_pytree(dx_dt)[0]

    # t = jnp.linspace(0.0, dt, 2)
    # _ode_flp = lambda t, x: _ode(x, t)
    # _dist_sqrt = odeint(_ode_flp, _dist_sqrt, t)[-1, :]

    _dist_sqrt = integrator(func=_ode, tk=0.0, yk=_dist_sqrt, dt=step_size)
    return _unflatten(_dist_sqrt)


def integrate_ode(
    prior_sqrt: MVNSqrt,
    observation: jnp.ndarray,
    observation_model: ConditionalMVNSqrt,
    sigma_points: Callable,
    integrator: Callable,
    step_size: float,
    criterion: Callable,
):
    def fun_to_iter(dist_sqrt):
        return ode_step(
            dist_sqrt,
            prior_sqrt,
            observation,
            observation_model,
            sigma_points,
            integrator,
            step_size,
        )

    return fixed_point(fun_to_iter, prior_sqrt, criterion)


def wasserstein_filter_sqrt(
    observations: jnp.ndarray,
    initial_dist: MVNSqrt,
    transition_model: ConditionalMVNSqrt,
    observation_model: ConditionalMVNSqrt,
    sigma_points: Callable,
    integrator: Callable = euler_odeint,
    step_size: float = 1e-2,
    stopping_criterion: Callable = lambda i, *_: i < 500,
):
    def body(carry, args):
        x, ell = carry
        y = args

        # predict
        F, b, Q_sqrt = linearize(transition_model, x)
        xp = predict(F, b, Q_sqrt, x)

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
        mu, sigma_sqrt = xp
        z, w = sigma_points(mu, sigma_sqrt)
        log_pdfs = jax.vmap(observation_model.logpdf, in_axes=(0, None))(z, y)
        ell += jnp.log(jnp.average(jnp.exp(log_pdfs), weights=w))

        return (xf, ell), xf

    x0 = initial_dist
    ys = observations

    (_, ell), Xf = jax.lax.scan(body, (x0, 0.0), xs=ys)
    Xf = none_or_concat(Xf, x0, 1)
    return Xf, ell
