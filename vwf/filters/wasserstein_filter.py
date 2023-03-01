from typing import Callable

import jax
import jax.random
from jax.flatten_util import ravel_pytree

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal as mvn

from vwf.objects import MVNStandard, ConditionalModel
from vwf.utils import euler, rk4, fixed_point, kullback_leibler


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
    return - mvn.logpdf(y, mean_y, cov_y) - mvn.logpdf(x, m, P)


def ode(dist, prior, obs, rv, obs_mdl, dt):
    n, d = rv.shape
    gradV = jax.grad(potential)

    _dist, _unravel_fcn = ravel_pytree(dist)

    def _ode_fcn(t, x, rv, d):
        mu, sigma = _unravel_fcn(x)

        z = mu + jnp.einsum("kh,nk->nh", jnp.linalg.cholesky(sigma), rv)
        dV = jax.vmap(gradV, in_axes=(0, None, None, None))(z, obs, prior, obs_mdl)

        mu_dt = - jnp.mean(dV, axis=0)
        sigma_dt = 2.0 * jnp.eye(d)\
                   - jnp.mean(jnp.einsum("nk,nh->nkh", dV, z - mu)
                              + jnp.einsum("nk,nh->nkh", z - mu, dV), axis=0)

        dx_dt = MVNStandard(mu_dt, sigma_dt)
        return ravel_pytree(dx_dt)[0]

    _dist = euler(_ode_fcn, 0.0, _dist, dt, rv=rv, d=d)
    # _dist = rk4(_ode_fcn, 0.0, _dist, dt, rv=rv, d=d)
    return _unravel_fcn(_dist)


def integrate_ode(prior: MVNStandard,
                  observation: jnp.ndarray,
                  noise: jnp.ndarray,
                  observation_model: ConditionalModel,
                  step_size: float, criterion: Callable):

    def fun_to_iter(dist):
        return ode(dist, prior,
                   observation, noise,
                   observation_model, step_size)
    return fixed_point(fun_to_iter, prior, criterion)


def wasserstein_filter(observations, random_vector,
                       init_dist, transition_model, observation_model,
                       step_size, stopping_criterion=kullback_leibler):

    def _ell(x, y, obs_mdl):
        mean_fcn, cov_fcn = obs_mdl
        mean_y, cov_y = mean_fcn(x), cov_fcn(x)
        return mvn.logpdf(y, mean_y, cov_y)

    def body(carry, args):
        x, ell = carry
        y, q = args

        # predict
        F, b, Q = linearize(transition_model, x)
        xp = predict(F, b, Q, x)

        # innovate
        xf = integrate_ode(xp, y, q,
                           observation_model,
                           step_size,
                           stopping_criterion)

        # ell
        mu, sigma = xp
        z = mu + jnp.einsum("kh,nk->nh", jnp.linalg.cholesky(sigma), q)
        log_wn = jax.vmap(_ell, in_axes=(0, None, None))(z, y, observation_model)
        ell += logsumexp(log_wn) - jnp.log(q.shape[0])

        return (xf, ell), xf

    x0 = init_dist
    y = observations
    q = random_vector

    (_, ell), Xf = jax.lax.scan(body, (x0, 0.0), xs=(y, q))
    return Xf, ell
