from typing import Callable

import jax
import jax.random
from jax.flatten_util import ravel_pytree

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal as mvn

from wasserstein_filter.objects import GMMSqrt, ConditionalMVNSqrt
from wasserstein_filter.utils import euler_odeint
from wasserstein_filter.utils import tria_qr, tria_tril
from wasserstein_filter.utils import none_or_concat


def linearize(model: ConditionalMVNSqrt, x: GMMSqrt):
    mean_fcn, cov_sqrt_fcn = model
    ms, _ = x

    Fs = jax.vmap(jax.jacfwd(mean_fcn))(ms)
    bs = jax.vmap(mean_fcn)(ms) - jnp.einsum("kij,kj->ki", Fs, ms)
    Qs_sqrt = jax.vmap(cov_sqrt_fcn)(ms)
    return Fs, bs, Qs_sqrt


def predict(Fs, bs, Qs_sqrt, x):
    ms, Ps_sqrt = x
    ms = jnp.einsum("kij,kj->ki", Fs, ms) + bs

    def _tria_fcn(F, P_sqrt, Q_sqrt):
        return tria_qr(jnp.concatenate([F @ P_sqrt, Q_sqrt], axis=1))

    Ps_sqrt = jax.vmap(_tria_fcn)(Fs, Ps_sqrt, Qs_sqrt)
    return GMMSqrt(ms, Ps_sqrt)


def log_posterior(
    state: jnp.ndarray,
    observation: jnp.ndarray,
    prior_sqrt: GMMSqrt,
    observation_model: ConditionalMVNSqrt,
):
    k = prior_sqrt.size

    ms, Ps_sqrt = prior_sqrt
    Ps = jnp.einsum("kij,knj->kin", Ps_sqrt, Ps_sqrt)
    prior_logpdf = logsumexp(mvn.logpdf(state, ms, Ps))

    obsrv_logpdf = observation_model.logpdf(state, observation)
    return obsrv_logpdf + prior_logpdf


def ode_step(
    key: jax.random.PRNGKey,
    dist_sqrt: GMMSqrt,
    prior_sqrt: GMMSqrt,
    observation: jnp.ndarray,
    observation_model: ConditionalMVNSqrt,
    monte_carlo_points: Callable,
    ode_integrator: Callable,
    step_size: float,
):
    d = dist_sqrt.dim
    k = dist_sqrt.size

    def log_ratio(state, mus, sigmas_sqrt):
        _log_target = log_posterior(state, observation, prior_sqrt, observation_model)

        sigmas_x = jnp.einsum("kij,knj->kin", sigmas_sqrt, sigmas_sqrt)
        log_mixture = logsumexp(mvn.logpdf(state, mus, sigmas_x)) - jnp.log(k)
        return _log_target - log_mixture

    gradP = jax.grad(log_ratio)
    hessP = jax.hessian(log_ratio)

    _dist_sqrt, _unflatten = ravel_pytree(dist_sqrt)

    key, sub_key = jax.random.split(key, 2)

    def _ode(t, x):
        mus, sigmas_sqrt = _unflatten(x)

        zs, ws = monte_carlo_points(sub_key, mus, sigmas_sqrt)

        def _grad_fcn(zs, mus, sigmas_sqrt):
            return jax.vmap(gradP, in_axes=(0, None, None))(
                zs, mus, sigmas_sqrt
            )

        dP = jax.vmap(_grad_fcn, in_axes=(0, None, None))(zs, mus, sigmas_sqrt)

        def _sigma_dt_fcn(z, w, mu, dP):
            return (
                jnp.einsum("ni,nj,n->ij", dP, z - mu, w)
                + jnp.einsum("ni,nj,n->ij", z - mu, dP, w)
            )

        sigmas_dt = jax.vmap(_sigma_dt_fcn)(zs, ws, mus, dP)

        def _sigma_sqrt_dt_fcn(sigma_sqrt, sigma_dt):
            sigma_sqrt_inv = jnp.linalg.inv(sigma_sqrt)
            return sigma_sqrt @ tria_tril(
                sigma_sqrt_inv @ sigma_dt @ sigma_sqrt_inv.T
            )

        mus_dt = jnp.einsum("kni,kn->ki", dP, ws)
        sigmas_sqrt_dt = jax.vmap(_sigma_sqrt_dt_fcn)(sigmas_sqrt, sigmas_dt)

        dx_dt = GMMSqrt(mus_dt, sigmas_sqrt_dt)
        return ravel_pytree(dx_dt)[0]

    _dist_sqrt = ode_integrator(func=_ode, tk=0.0, yk=_dist_sqrt, dt=step_size)
    return key, _unflatten(_dist_sqrt)


def integrate_ode(
    key: jax.random.PRNGKey,
    prior_sqrt: GMMSqrt,
    observation: jnp.ndarray,
    observation_model: ConditionalMVNSqrt,
    monte_carlo_points: Callable,
    ode_integrator: Callable,
    step_size: float,
    nb_iter: int
):
    def fun_to_iter(carry, args):
        key, dist_sqrt = carry
        _ = args
        return ode_step(
            key,
            dist_sqrt,
            prior_sqrt,
            observation,
            observation_model,
            monte_carlo_points,
            ode_integrator,
            step_size,
        ), _

    return jax.lax.scan(fun_to_iter, (key, prior_sqrt), xs=jnp.arange(nb_iter))[0]


def wasserstein_filter_sqrt_gmm(
    key: jax.random.PRNGKey,
    observations: jnp.ndarray,
    initial_dist: GMMSqrt,
    transition_model: ConditionalMVNSqrt,
    observation_model: ConditionalMVNSqrt,
    monte_carlo_points: Callable,
    ode_integrator: Callable = euler_odeint,
    step_size: float = 1e-2,
    nb_iter: int = 500
):
    def body(carry, args):
        key, x, ell = carry
        y = args

        # predict
        Fs, bs, Qs_sqrt = linearize(transition_model, x)
        xp = predict(Fs, bs, Qs_sqrt, x)

        # innovate
        key, xf = integrate_ode(
            key,
            xp,
            y,
            observation_model,
            monte_carlo_points,
            ode_integrator,
            step_size,
            nb_iter,
        )

        # ell
        mus, sigmas_sqrt = xp
        key, sub_key = jax.random.split(key, 2)
        zs, ws = monte_carlo_points(sub_key, mus, sigmas_sqrt)

        def _ell(z, w):
            _log_pdfs = jax.vmap(observation_model.logpdf, in_axes=(0, None))(z, y)
            return jnp.log(jnp.average(jnp.exp(_log_pdfs), weights=w))

        ell += jnp.mean(jax.vmap(_ell)(zs, ws))
        return (key, xf, ell), xf

    x0 = initial_dist
    ys = observations

    (_, _, ell), Xf = jax.lax.scan(body, (key, x0, 0.0), xs=ys)
    Xf = none_or_concat(Xf, x0, 1)
    return Xf, ell
