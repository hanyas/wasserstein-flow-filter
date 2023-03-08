from typing import Callable

import jax
import jax.random
from jax.flatten_util import ravel_pytree

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal as mvn
from jax.experimental.ode import odeint

from vwf.objects import MVNSqrt, ConditionalModelSqrt
from vwf.utils import fixed_point, rk4_odeint, euler_odeint
from vwf.utils import kullback_leibler_mvn_sqrt_cond, wasserstein_mvn_sqrt_cond
from vwf.utils import tria_qr, tria_tril


def linearize(model: ConditionalModelSqrt, x: MVNSqrt):
    mean, cov_sqrt = model
    m, _ = x

    F = jax.jacfwd(mean, 0)(m)
    b = mean(m) - F @ m
    Q_sqrt = cov_sqrt(m)
    return F, b, Q_sqrt


def predict(F, b, Q_sqrt, x):
    m, P_sqrt = x
    m = F @ m + b
    P_sqrt = tria_qr(jnp.concatenate([F @ P_sqrt, Q_sqrt], axis=1))
    return MVNSqrt(m, P_sqrt)


def potential(x, y, prior_sqrt, obs_mdl):
    m, P_sqrt = prior_sqrt
    mean_fcn, cov_sqrt_fcn = obs_mdl
    mean_y, cov_sqrt_y = mean_fcn(x), cov_sqrt_fcn(x)
    return (
        - mvn.logpdf(y, mean_y, cov_sqrt_y @ cov_sqrt_y.T)
        - mvn.logpdf(x, m, P_sqrt @ P_sqrt.T)
    )


def ode(
    dist_sqrt: MVNSqrt,
    prior_sqrt: MVNSqrt,
    observation: jnp.ndarray,
    observation_model: ConditionalModelSqrt,
    sigma_points: Callable,
    integrator: Callable,
    step_size: float,
):
    d = dist_sqrt.dim
    gradV = jax.grad(potential)

    _dist_sqrt, _unflatten = ravel_pytree(dist_sqrt)

    def _ode_fcn(t, x):
        mu, sigma_sqrt = _unflatten(x)

        z, w = sigma_points(mu, sigma_sqrt)

        dV = jax.vmap(gradV, in_axes=(0, None, None, None))(
            z, observation, prior_sqrt, observation_model
        )

        sigma_dt = (
            2.0 * jnp.eye(d)
            - jnp.einsum("nk,nh,n->kh", dV, z - mu, w)
            - jnp.einsum("nk,nh,n->kh", z - mu, dV, w)
        )

        sigma_sqrt_inv = jnp.linalg.inv(sigma_sqrt)

        mu_dt = -jnp.einsum("nk,n->k", dV, w)
        sigma_sqrt_dt = sigma_sqrt @ tria_tril(
            sigma_sqrt_inv @ sigma_dt @ sigma_sqrt_inv.T
        )

        dx_dt = MVNSqrt(mu_dt, sigma_sqrt_dt)
        return ravel_pytree(dx_dt)[0]

    # t = jnp.linspace(0.0, dt, 2)
    # _ode_fcn_flp = lambda t, x: _ode_fcn(x, t)
    # _dist_sqrt = odeint(_ode_fcn_flp, _dist_sqrt, t)[-1, :]

    _dist_sqrt = integrator(_ode_fcn, 0.0, _dist_sqrt, dt=step_size)
    return _unflatten(_dist_sqrt)


def integrate_ode(
    prior_sqrt: MVNSqrt,
    observation: jnp.ndarray,
    observation_model: ConditionalModelSqrt,
    sigma_points: Callable,
    integrator: Callable,
    step_size: float,
    criterion: Callable,
):
    def fun_to_iter(dist_sqrt):
        return ode(
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
    transition_model: ConditionalModelSqrt,
    observation_model: ConditionalModelSqrt,
    sigma_points: Callable,
    integrator: Callable = euler_odeint,
    step_size: float = 1e-2,
    stopping_criterion: Callable = kullback_leibler_mvn_sqrt_cond,
):
    def _cond_log_pdf(x, y, obs_mdl):
        mean_fcn, cov_sqrt_fcn = obs_mdl
        mean_y, cov_sqrt_y = mean_fcn(x), cov_sqrt_fcn(x)
        return mvn.logpdf(y, mean_y, cov_sqrt_y @ cov_sqrt_y.T)

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
        log_pdfs = jax.vmap(_cond_log_pdf, in_axes=(0, None, None))(
            z, y, observation_model
        )
        ell += jnp.log(jnp.average(jnp.exp(log_pdfs), weights=w))

        return (xf, ell), xf

    x0 = initial_dist
    ys = observations

    (_, ell), xf = jax.lax.scan(body, (x0, 0.0), xs=ys)
    return xf, ell
