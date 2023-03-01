from typing import Callable

import jax
import jax.random
from jax.flatten_util import ravel_pytree

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal as mvn

from vwf.objects import MVNSqrt, ConditionalModelSqrt
from vwf.utils import euler, rk4, fixed_point, kullback_leibler_sqrt
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
    return - mvn.logpdf(y, mean_y, cov_sqrt_y @ cov_sqrt_y.T)\
        - mvn.logpdf(x, m, P_sqrt @ P_sqrt.T)


def ode(dist_sqrt, prior_sqrt, obs, rv, obs_mdl, dt):
    n, d = rv.shape
    gradV = jax.grad(potential)

    _dist_sqrt, _unravel_fcn = ravel_pytree(dist_sqrt)

    def _ode_fcn(t, x, rv, d):
        mu, sigma_sqrt = _unravel_fcn(x)

        z = mu + jnp.einsum("kh,nk->nh", sigma_sqrt, rv)
        dV = jax.vmap(gradV, in_axes=(0, None, None, None))(z, obs, prior_sqrt, obs_mdl)

        sigma_dt = 2.0 * jnp.eye(d)\
                   - jnp.mean(jnp.einsum("nk,nh->nkh", dV, z - mu)
                              + jnp.einsum("nk,nh->nkh", z - mu, dV), axis=0)

        sigma_sqrt_inv = jnp.linalg.inv(sigma_sqrt)

        mu_dt = - jnp.mean(dV, axis=0)
        sigma_sqrt_dt = sigma_sqrt @ tria_tril(sigma_sqrt_inv @ sigma_dt @ sigma_sqrt_inv.T)

        dx_dt = MVNSqrt(mu_dt, sigma_sqrt_dt)
        return ravel_pytree(dx_dt)[0]

    _dist_sqrt = euler(_ode_fcn, 0.0, _dist_sqrt, dt, rv=rv, d=d)
    # _dist_sqrt = rk4(_ode_fcn, 0.0, _dist_sqrt, dt, rv=rv, d=d)
    return _unravel_fcn(_dist_sqrt)


def integrate_ode(prior_sqrt: MVNSqrt,
                  observation: jnp.ndarray,
                  noise: jnp.ndarray,
                  observation_model: ConditionalModelSqrt,
                  step_size: float, criterion: Callable):

    def fun_to_iter(dist_sqrt):
        return ode(dist_sqrt, prior_sqrt,
                   observation, noise,
                   observation_model, step_size)
    return fixed_point(fun_to_iter, prior_sqrt, criterion)


def wasserstein_filter_sqrt(observations, random_vector,
                            init_dist, transition_model, observation_model,
                            step_size, stopping_criterion=kullback_leibler_sqrt):

    def _ell(x, y, obs_mdl):
        mean_fcn, cov_sqrt_fcn = obs_mdl
        mean_y, cov_sqrt_y = mean_fcn(x), cov_sqrt_fcn(x)
        return mvn.logpdf(y, mean_y, cov_sqrt_y @ cov_sqrt_y.T)

    def body(carry, args):
        x, ell = carry
        y, q = args

        # predict
        F, b, Q_sqrt = linearize(transition_model, x)
        xp = predict(F, b, Q_sqrt, x)

        # innovate
        xf = integrate_ode(xp, y, q,
                           observation_model,
                           step_size,
                           stopping_criterion)

        # ell
        mu, sigma_sqrt = xp
        z = mu + jnp.einsum("kh,nk->nh", sigma_sqrt, q)
        log_wn = jax.vmap(_ell, in_axes=(0, None, None))(z, y, observation_model)
        ell += logsumexp(log_wn) - jnp.log(q.shape[0])

        return (xf, ell), xf

    x0 = init_dist
    y = observations
    q = random_vector

    (_, ell), Xf = jax.lax.scan(body, (x0, 0.0), xs=(y, q))
    return Xf, ell
