from typing import Callable

import jax
import jax.random
from jax.flatten_util import ravel_pytree

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal as mvn
from jax.experimental.ode import odeint

from vwf.objects import GMMSqrt, MVNSqrt, ConditionalMVNSqrt
from vwf.utils import fixed_point, rk4_odeint, euler_odeint
from vwf.utils import kullback_leibler_mvn_sqrt_cond, wasserstein_mvn_sqrt_cond
from vwf.utils import tria_qr, tria_tril


def linearize(model: ConditionalMVNSqrt, x: GMMSqrt):
    mean_fcn, cov_sqrt_fcn = model
    ms, _ = x

    Fs = jax.vmap(jax.jacfwd(mean_fcn))(ms)
    bs = jax.vmap(mean_fcn)(ms) - jnp.einsum("nij,nj->ni", Fs, ms)
    Qs_sqrt = jax.vmap(cov_sqrt_fcn)(ms)
    return Fs, bs, Qs_sqrt


def predict(Fs, bs, Qs_sqrt, x):
    ms, Ps_sqrt = x
    ms = jnp.einsum("nij,nj->ni", Fs, ms) + bs

    def _tria_fcn(F, P_sqrt, Q_sqrt):
        return tria_qr(jnp.concatenate([F @ P_sqrt, Q_sqrt], axis=1))

    Ps_sqrt = jax.vmap(_tria_fcn)(Fs, Ps_sqrt, Qs_sqrt)
    return GMMSqrt(ms, Ps_sqrt)


def log_target(
    state: jnp.ndarray,
    observation: jnp.ndarray,
    prior_sqrt: GMMSqrt,
    observation_model: ConditionalMVNSqrt,
):
    ms, Ps_sqrt = prior_sqrt
    mean_fcn, cov_sqrt_fcn = observation_model
    means_y = mean_fcn(state)
    covs_sqrt_y = cov_sqrt_fcn(state)

    covs_y = covs_sqrt_y @ covs_sqrt_y.T
    covs_x = jnp.einsum("nij,nkj->nik", Ps_sqrt, Ps_sqrt)
    out = (
        mvn.logpdf(observation, means_y, covs_y)
        + mvn.logpdf(state, ms, covs_x)
    )
    return logsumexp(out)


def ode_step(
    dist_sqrt: GMMSqrt,
    prior_sqrt: GMMSqrt,
    observation: jnp.ndarray,
    observation_model: ConditionalMVNSqrt,
    sigma_points: Callable,
    integrator: Callable,
    step_size: float,
):
    d = dist_sqrt.dim
    k = dist_sqrt.size

    def log_ratio(state, mus, sigmas_sqrt):
        _log_target = log_target(
            state, observation, prior_sqrt, observation_model
        )

        sigmas_x = jnp.einsum("nij,nkj->nik", sigmas_sqrt, sigmas_sqrt)
        log_mixture = logsumexp(mvn.logpdf(state, mus, sigmas_x))
        return _log_target - log_mixture

    gradV = jax.grad(log_ratio)
    hessV = jax.hessian(log_ratio)

    _dist_sqrt, _unflatten = ravel_pytree(dist_sqrt)

    def _ode(t, x):
        mus, sigmas_sqrt = _unflatten(x)

        zs, ws = jax.vmap(sigma_points)(mus, sigmas_sqrt)

        def _grad_fcn(zs, mus, sigmas_sqrt):
            return jax.vmap(gradV, in_axes=(0, None, None))(
                zs, mus, sigmas_sqrt
            )

        dV = jax.vmap(_grad_fcn, in_axes=(0, None, None))(zs, mus, sigmas_sqrt)

        def _sigma_dt_fcn(z, w, mu, dV):
            return (
                jnp.einsum("ni,nj,n->ij", dV, z - mu, w)
                + jnp.einsum("ni,nj,n->ij", z - mu, dV, w)
            )

        sigmas_dt = jax.vmap(_sigma_dt_fcn)(zs, ws, mus, dV)

        def _sigma_sqrt_dt_fcn(sigma_sqrt, sigma_dt):
            sigma_sqrt_inv = jnp.linalg.inv(sigma_sqrt)
            return sigma_sqrt @ tria_tril(
                sigma_sqrt_inv @ sigma_dt @ sigma_sqrt_inv.T
            )

        mus_dt = jnp.einsum("kni,kn->ki", dV, ws)
        sigmas_sqrt_dt = jax.vmap(_sigma_sqrt_dt_fcn)(sigmas_sqrt, sigmas_dt)

        dx_dt = GMMSqrt(mus_dt, sigmas_sqrt_dt)
        return ravel_pytree(dx_dt)[0]

    _dist_sqrt = integrator(func=_ode, tk=0.0, yk=_dist_sqrt, dt=step_size)
    return _unflatten(_dist_sqrt)


def integrate_ode(
    prior_sqrt: GMMSqrt,
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


def wasserstein_filter_sqrt_gmm(
    observations: jnp.ndarray,
    initial_dist: GMMSqrt,
    transition_model: ConditionalMVNSqrt,
    observation_model: ConditionalMVNSqrt,
    sigma_points: Callable,
    integrator: Callable = euler_odeint,
    step_size: float = 1e-2,
    stopping_criterion: Callable = lambda i, *_: i < 500,
):
    def _cond_log_pdf(x, y, obs_mdl):
        mean_fcn, cov_sqrt_fcn = obs_mdl
        mean_y, cov_sqrt_y = mean_fcn(x), cov_sqrt_fcn(x)
        return mvn.logpdf(y, mean_y, cov_sqrt_y @ cov_sqrt_y.T)

    def body(carry, args):
        x, ell = carry
        y = args

        # predict
        Fs, bs, Qs_sqrt = linearize(transition_model, x)
        xp = predict(Fs, bs, Qs_sqrt, x)

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
        mus, sigmas_sqrt = xp
        zs, ws = jax.vmap(sigma_points)(mus, sigmas_sqrt)

        def _ell(z, w):
            _log_pdfs = jax.vmap(_cond_log_pdf, in_axes=(0, None, None))(
                z, y, observation_model
            )
            return jnp.log(jnp.average(jnp.exp(_log_pdfs), weights=w))

        ell += jnp.mean(jax.vmap(_ell)(zs, ws))
        return (xf, ell), xf

    x0 = initial_dist
    ys = observations

    (_, ell), xf = jax.lax.scan(body, (x0, 0.0), xs=ys)
    return xf, ell
