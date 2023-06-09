import jax
import jax.numpy as jnp
import jax.random

from jax.scipy.linalg import cho_solve
from jax.scipy.stats import multivariate_normal as mvn

from wasserstein_filter.objects import MVNStandard, ConditionalMVN
from wasserstein_filter.utils import none_or_concat


def kalman_filter(
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: ConditionalMVN,
    observation_model: ConditionalMVN,
):
    def _linearize(model, x):
        mean, cov = model
        m, p = x

        F = jax.jacfwd(mean, 0)(m)
        b = mean(m) - F @ m
        Q = cov(m)
        return F, b, Q

    def _predict(F, b, Q, x):
        m, P = x
        m = F @ m + b
        P = Q + F @ P @ F.T
        return MVNStandard(m, P)

    def _update(H, c, R, x, y):
        m, P = x

        y_hat = H @ m + c
        S = R + H @ P @ H.T
        chol_S = jnp.linalg.cholesky(S)
        G = P @ cho_solve((chol_S, True), H).T

        m = m + G @ (y - y_hat)
        P = P - G @ S @ G.T
        ell = mvn.logpdf(y, y_hat, chol_S @ chol_S.T)
        return MVNStandard(m, P), ell

    def body(carry, args):
        xf, ell = carry
        y = args

        F_x, b, Q = _linearize(transition_model, xf)
        xp = _predict(F_x, b, Q, xf)

        H_x, c, R = _linearize(observation_model, xp)
        xi, _ell = _update(H_x, c, R, xp, y)
        return (xi, ell + _ell), xi

    x0 = initial_dist
    (_, ell), Xs = jax.lax.scan(body, (x0, 0.0), observations)

    Xs = none_or_concat(Xs, x0, 1)
    return Xs, ell
