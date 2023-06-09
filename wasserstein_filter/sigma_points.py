import math

import jax
import jax.numpy as jnp
import numpy as np


def monte_carlo_points(key, dim, nb_samples):
    key, sub_key = jax.random.split(key, 2)
    rv = jax.random.normal(sub_key, shape=(nb_samples, dim))
    wm = jnp.ones((nb_samples, )) / nb_samples
    return key, rv, wm


def cubature_points(mu, cov_sqrt):
    n_dim = mu.shape[0]
    xi, wm = _cubature_weights(n_dim)
    p = mu[None, :] + jnp.dot(cov_sqrt, xi.T).T
    return p, wm


def _cubature_weights(n_dim: int):
    wm = np.ones(shape=(2 * n_dim,)) / (2 * n_dim)
    xi = np.concatenate([np.eye(n_dim), -np.eye(n_dim)], axis=0) * np.sqrt(n_dim)
    return xi, wm


def gauss_hermite_points(mu, cov_sqrt, order: int):
    n_dim = mu.shape[0]
    xi, wm = _gauss_hermite_weights(n_dim, order)
    p = mu[None, :] + math.sqrt(2) * (cov_sqrt @ xi).T
    return p, wm


def _gauss_hermite_weights(n_dim: int, order: int = 3):
    n = n_dim
    p = order

    hermite_coeff = _hermite_coeff(p)
    hermite_roots = np.flip(np.roots(hermite_coeff[-1]))

    table = np.zeros(shape=(n, p ** n))

    w_1d = np.zeros(shape=(p,))
    for i in range(p):
        w_1d[i] = (2 ** (p - 1) * np.math.factorial(p) * np.sqrt(np.pi) /
                   (p ** 2 * (np.polyval(hermite_coeff[p - 1],
                                         hermite_roots[i])) ** 2))

    # Get roll table
    for i in range(n):
        base = np.ones(shape=(1, p ** (n - i - 1)))
        for j in range(1, p):
            base = np.concatenate([base,
                                   (j + 1) * np.ones(shape=(1, p ** (n - i - 1)))],
                                  axis=1)
        table[n - i - 1, :] = np.tile(base, (1, int(p ** i)))

    table = table.astype("int64") - 1

    s = 1 / (np.sqrt(np.pi) ** n)

    wm = s * np.prod(w_1d[table], axis=0)
    xi = hermite_roots[table]

    return xi, wm


def _hermite_coeff(order: int):
    H0 = np.array([1])
    H1 = np.array([2, 0])

    H = [H0, H1]
    for i in range(2, order + 1):
        H.append(2 * np.append(H[i - 1], 0) -
                 2 * (i - 1) * np.pad(H[i - 2], (2, 0), 'constant', constant_values=0))

    return H
