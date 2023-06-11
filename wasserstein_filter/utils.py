from functools import partial

import jax
from jax import closure_convert, custom_vjp, vjp, numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.lax import while_loop

from wasserstein_filter.objects import MVNStandard, MVNSqrt


logdet = lambda x: jnp.linalg.slogdet(x)[1]


def symmetrize(A):
    return 0.5 * (A + A.T)


def wasserstein_mvn_cond(i, q, p):
    return wasserstein_mvn(q, p) > 1e-8


def wasserstein_mvn_sqrt_cond(i, q, p):
    return wasserstein_mvn_sqrt(q, p) > 1e-8


def kullback_leibler_mvn_cond(i, q, p):
    return kullback_leibler_mvn(q, p) > 1e-8


def kullback_leibler_mvn_sqrt_cond(i, q, p):
    return kullback_leibler_mvn_sqrt(q, p) > 1e-8


def wasserstein_mvn(q: MVNStandard, p: MVNStandard):
    x, X = q
    y, Y = p

    from jax.scipy.linalg import sqrtm

    return jnp.linalg.norm(x - y) ** 2 + jnp.trace(
        X + Y - 2.0 * sqrtm(sqrtm(Y) @ X @ sqrtm(Y))
    )


def wasserstein_mvn_sqrt(q: MVNSqrt, p: MVNSqrt):
    x, X_sqrt = q
    y, Y_sqrt = p

    X = symmetrize(X_sqrt @ X_sqrt.T)
    Y = symmetrize(Y_sqrt @ Y_sqrt.T)

    Y_chol = jnp.linalg.cholesky(Y)
    return jnp.linalg.norm(x - y) ** 2 + jnp.trace(
        X + Y
        - 2.0 * jnp.linalg.cholesky(Y_chol @ X @ Y_chol)
    )


def kullback_leibler_mvn(q: MVNStandard, p: MVNStandard):
    # KL(p || q)
    x, X = q
    y, Y = p
    d = q.dim

    X_inv = jnp.linalg.inv(X)
    return (
        jnp.trace(X_inv @ Y) / 2.0
        - d / 2.0
        + jnp.dot(x - y, jnp.dot(X_inv, x - y)) / 2.0
        + (logdet(X) - logdet(Y)) / 2.0
    )


def kullback_leibler_mvn_sqrt(q: MVNSqrt, p: MVNSqrt):
    # KL(p || q)
    x, X_sqrt = q
    y, Y_sqrt = p
    d = q.dim

    X_sqrt_inv = jnp.linalg.inv(X_sqrt)
    X_inv = X_sqrt_inv.T @ X_sqrt_inv
    Y = Y_sqrt @ Y_sqrt.T
    return (
        jnp.trace(X_inv @ Y) / 2.0
        - d / 2.0
        + jnp.dot(x - y, jnp.dot(X_inv, x - y)) / 2.0
        + jnp.sum(jnp.log(jnp.diag(X_sqrt)))
        - jnp.sum(jnp.log(jnp.diag(Y_sqrt)))
    )


def rk4_odeint(func, tk, yk, dt):
    f1 = func(tk, yk)
    f2 = func(tk + dt / 2.0, yk + (f1 * (dt / 2.0)))
    f3 = func(tk + dt / 2.0, yk + (f2 * (dt / 2.0)))
    f4 = func(tk + dt, yk + (f3 * dt))
    return yk + (dt / 6.0) * (f1 + (2.0 * f2) + (2.0 * f3) + f4)


def euler_odeint(func, tk, yk, dt):
    return yk + dt * func(tk, yk)


def fixed_point(f, x0, criterion):
    converted_fn, aux_args = closure_convert(f, x0)
    return _fixed_point(converted_fn, aux_args, x0, criterion)


@partial(custom_vjp, nondiff_argnums=(0, 3))
def _fixed_point(f, params, x0, criterion):
    return __fixed_point(f, params, x0, criterion)[0]


def _fixed_point_fwd(f, params, x0, criterion):
    x_star, n_iter = __fixed_point(f, params, x0, criterion)
    return x_star, (params, x_star, n_iter)


def _fixed_point_rev(f, _criterion, res, x_star_bar):
    params, x_star, n_iter = res
    _, vjp_theta = vjp(lambda p: f(x_star, *p), params)
    (theta_bar,) = vjp_theta(
        __fixed_point(
            partial(_rev_iter, f),
            (params, x_star, x_star_bar),
            x_star_bar,
            lambda i, *_: i < n_iter + 1,
        )[0]
    )
    return theta_bar, jax.tree_map(jnp.zeros_like, x_star)


def _rev_iter(f, u, *packed):
    params, x_star, x_star_bar = packed
    _, vjp_x = vjp(lambda x: f(x, *params), x_star)
    ravelled_x_star_bar, unravel_fn = ravel_pytree(x_star_bar)
    ravelled_vjp_x_u, _ = ravel_pytree(vjp_x(u)[0])
    return unravel_fn(ravelled_x_star_bar + ravelled_vjp_x_u)


def __fixed_point(f, params, x0, criterion):
    def cond_fun(carry):
        i, x_prev, x = carry
        return criterion(i, x_prev, x)

    def body_fun(carry):
        i, _, x = carry
        return i + 1, x, f(x, *params)

    n_iter, _, x_star = while_loop(cond_fun, body_fun, (1, x0, f(x0, *params)))
    return x_star, n_iter


_fixed_point.defvjp(_fixed_point_fwd, _fixed_point_rev)


def tria_tril(A):
    L = jnp.tril(A, k=-1)
    return L + jnp.diag(jnp.diag(A)) / 2.0


def tria_qr(A):
    return qr(A.T).T


@jax.custom_jvp
def qr(A: jnp.ndarray):
    """The JAX provided implementation is not parallelizable using VMAP.
    As a consequence, we have to rewrite it..."""
    return _qr(A)


def _qr(A: jnp.ndarray, return_q=False):
    m, n = A.shape
    min_ = min(m, n)
    if return_q:
        Q = jnp.eye(m)

    for j in range(min_):
        # Apply Householder transformation.
        v, tau = _householder(A[j:, j])

        H = jnp.eye(m)
        H = H.at[j:, j:].add(-tau * (v[:, None] @ v[None, :]))

        A = H @ A
        if return_q:
            Q = H @ Q  # noqa

    R = jnp.triu(A[:min_, :min_])
    if return_q:
        return Q[:n].T, R  # noqa
    else:
        return R


def _householder(a):
    if a.dtype == jnp.float64:
        eps = 1e-9
    else:
        eps = 1e-7

    alpha = a[0]
    s = jnp.sum(a[1:] ** 2)
    cond = s < eps

    def if_not_cond(v):
        t = (alpha**2 + s) ** 0.5
        v0 = jax.lax.cond(
            alpha <= 0, lambda _: alpha - t, lambda _: -s / (alpha + t), None
        )
        tau = 2 * v0**2 / (s + v0**2)
        v = v / v0
        v = v.at[0].set(1.0)
        return v, tau

    return jax.lax.cond(cond, lambda v: (v, 0.0), if_not_cond, a)


def qr_jvp_rule(primals, tangents):
    (x,) = primals
    (dx,) = tangents
    q, r = _qr(x, True)
    m, n = x.shape
    min_ = min(m, n)
    if m < n:
        dx = dx[:, :m]
    dx_rinv = jax.lax.linalg.triangular_solve(r, dx)
    qt_dx_rinv = jnp.matmul(q.T, dx_rinv)
    qt_dx_rinv_lower = jnp.tril(qt_dx_rinv, -1)
    do = qt_dx_rinv_lower - qt_dx_rinv_lower.T  # This is skew-symmetric
    # The following correction is necessary for complex inputs
    do = do + jnp.eye(min_, dtype=do.dtype) * (
        qt_dx_rinv - jnp.real(qt_dx_rinv)
    )
    dr = jnp.matmul(qt_dx_rinv - do, r)
    return r, dr


qr.defjvp(qr_jvp_rule)


def none_or_concat(x, y, position=1):
    if x is None or y is None:
        return None
    if position == 1:
        return jax.tree_map(
            lambda a, b: jnp.concatenate([a[None, ...], b]), y, x
        )
    else:
        return jax.tree_map(
            lambda a, b: jnp.concatenate([b, a[None, ...]]), y, x
        )
