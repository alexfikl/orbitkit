# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.linalg as la

from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


def stringify_adjacency(mat: Array) -> str:
    symbols = {0: " ◻ ", 1: " ◼ "}

    return "\n".join(
        "".join(symbols[int(mat[i, j] != 0)] for j in range(mat.shape[1]))
        for i in range(mat.shape[0])
    )


# {{{ adjacency matrices


def generate_adjacency_all(n: int, *, dtype: Any = None) -> Array:
    r"""Generate a all-to-all :math:`n \times n` adjacency matrix."""
    if dtype is None:
        dtype = np.int32

    return np.ones((n, n), dtype=dtype)


def generate_adjacency_feed_forward(n: int, *, dtype: Any = None) -> Array:
    r"""Generate a :math:`n \times n` lower triangular adjacency matrix."""
    if dtype is None:
        dtype = np.int32

    result = np.ones((n, n), dtype=dtype)
    return np.tril(result, k=0)


def generate_adjacency_ring(n: int, *, k: int = 1, dtype: Any = None) -> Array:
    """Generate a *k*-ring network with :math:`n` nodes.

    In this network, each node is connected to its :math:`k` nearest neighbors
    with periodicity. For a non-periodic version see :func:`generate_adjacency_bus`.
    """
    if dtype is None:
        dtype = np.int32

    if not 0 <= k < n:
        raise ValueError(f"Number of neighbors 'm' is invalid: '{k}' (not in [0, {n})")

    # NOTE: this is essentially just a periodic banded matrix
    eye = np.eye(n, dtype=dtype)
    result = np.zeros((n, n), dtype=dtype)

    for i in range(-k, k + 1):
        result += np.roll(eye, i, axis=1)

    return result


def generate_adjacency_bus(n: int, *, k: int = 1, dtype: Any = None) -> Array:
    """Generate a bus network with :math:`n` nodes.

    In this network, each node is connected to its :math:`k` nearest neighbors
    in a non-periodic fashion.
    """
    if dtype is None:
        dtype = np.int32

    if not 0 <= k < n:
        raise ValueError(f"Number of neighbors 'm' is invalid: '{k}' (not in [0, {n}])")

    # NOTE: this is essentially just a non-periodic banded matrix
    ones = np.ones(n, dtype=dtype)
    result = np.zeros((n, n), dtype=dtype)

    for i in range(-k, k + 1):
        result += np.diag(ones[abs(i) :], k=i)

    return result


def generate_adjacency_star(n: int, *, dtype: Any = None) -> Array:
    """Generate a star network with :math:`n` nodes.

    In this network, there is a central node connected to all nodes.
    """
    if dtype is None:
        dtype = np.int32

    result = np.zeros((n, n), dtype=dtype)
    result[0, 1:] = 1
    result[1:, 0] = 1

    np.fill_diagonal(result, 1)
    return result


def generate_adjacency_star_tree(
    n: int, *, nhubs: int | None = None, dtype: Any = None
) -> Array:
    """Generate a star of stars network with :math:`n` nodes."""

    if nhubs is None:
        nhubs = n // 5

    if dtype is None:
        dtype = np.int32

    # NOTE: the way this is going to be constructed is
    # - 1 central node
    # - nhubs hub central nodes
    # - remaining nodes get divided between hubs
    # FIXME: this could be made nicely recursive for additional fanciness
    n_hub_nodes = n - nhubs - 1
    partitions = 1 + nhubs + np.linspace(0, n_hub_nodes, nhubs + 1, dtype=dtype)

    result = np.zeros((n, n), dtype=dtype)
    for m in range(nhubs):
        # connect root note to current hub
        result[0, m + 1] = result[m + 1, 0] = 1

        # connect nodes to hub
        leaves = np.s_[partitions[m] : partitions[m + 1]]
        result[m + 1, leaves] = result[leaves, m + 1] = 1

    np.fill_diagonal(result, 1)
    return result


def generate_adjacency_lattice(
    n: int,
    m: int | None = None,
    *,
    dtype: Any = None,
) -> Array:
    r"""Generate a lattice network with :math:`n` nodes.

    In this network, every node is connected to at most 4 other nodes in such a
    way that they form a two dimensional grid.

    The algorithm attempts to generate a grid close to :math:`\sqrt{n} \times
    \sqrt{n}`, so ensure that :math:`n` is not prime and can be factorized nicely.
    In the most degenerate case, this will create a bus network. If both :math:`n`
    and :math:`m` are given, then a :math:`n \times m` grid with :math:`n m` nodes
    is created.
    """
    if dtype is None:
        dtype = np.int32

    if m is None:
        m = int(np.sqrt(n)) + 1
        while n % m != 0:
            m -= 1

        assert n % m == 0

        n //= m

    Im = np.eye(m, dtype=dtype)
    Tm = generate_adjacency_bus(m, dtype=dtype)
    In = np.eye(n, dtype=dtype)
    Tn = generate_adjacency_bus(n, dtype=dtype)

    return np.kron(Im, Tn) + np.kron(Tm, In)


def generate_adjacency_erdos_renyi(
    n: int,
    *,
    p: float | None = None,
    k: int | None = None,
    dtype: Any = None,
    symmetric: bool = True,
    rng: np.random.Generator | None = None,
) -> Array:
    r"""Generate a random Erdős-Rényi :math:`n \times n` adjacency matrix.

    :arg p: probability of an edge between two nodes (defaults to *0.25*).
    :arg k: average number of edges for each node (i.e. the degree). If *p* is
        not given, it is computed as :math:`p = k / (n - 1)`.
    :arg symmetric: if *True*, the adjacency matrix will be symmetric.
    """
    if p is not None and k is not None:
        raise ValueError("cannot pass both 'p' and 'k'")

    if p is None:
        p = 0.25 if k is None else (k / (n - 1))

    if dtype is None:
        dtype = np.int32

    if rng is None:
        rng = np.random.default_rng()

    if symmetric:
        rows, cols = np.tril_indices(n)
        mask = rng.random(size=rows.size) < p

        result = np.zeros((n, n), dtype=dtype)
        result[rows[mask], cols[mask]] = 1
        result[cols[mask], rows[mask]] = 1
    else:
        result = (rng.random(size=(n, n)) < p).astype(dtype)

    return result


def generate_adjacency_strogatz_watts(
    n: int,
    *,
    k: int = 2,
    p: float = 0.1,
    dtype: Any = None,
    rng: np.random.Generator | None = None,
) -> Array:
    r"""Generate a random Strogatz-Watts :math:`n \times n` adjacency matrix.

    :arg k: number of neighboring nodes.
    :arg p: rewiring probability.
    """

    if dtype is None:
        dtype = np.int32

    if rng is None:
        rng = np.random.default_rng()

    result = generate_adjacency_ring(n, k=k, dtype=dtype)
    for i in range(n):
        forbidden = {(i + j) % n for j in range(-k, k + 1)}

        for j in range(1, k + 1):
            if not rng.random() < p:
                continue

            # remove current edge
            jold = (i + j) % n
            result[i, jold] = result[jold, i] = 0

            # rewire to a new edge
            choices = [c for c in range(n) if c not in forbidden]
            jnew = rng.choice(choices)
            result[i, jnew] = result[jnew, i] = 1

    return result


# }}}

# {{{ weights


def generate_random_weights(
    mat: Array,
    *,
    dtype: Any = None,
    symmetric: bool = False,
    rng: np.random.Generator | None = None,
) -> Array:
    """Generate a random weight matrix the same size as *mat*.

    :arg symmetric: if *True*, the weight matrix will be symmetric.
    """
    if rng is None:
        rng = np.random.default_rng()

    # NOTE: `mat` will likely have an integer dtype, so we cannot use it as a default

    # TODO: support more distributions
    w = rng.uniform(0.01, 1.0, size=mat.shape).astype(dtype)
    if symmetric:
        w = (w + w.T) / 2

    return mat * w  # type: ignore[no-any-return]


def generate_random_gaussian_weights(
    mat: Array,
    *,
    dtype: Any = None,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Array:
    """Generate a random weight matrix based on Gaussian node distance.

    :arg sigma: standard deviation of the normal distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    # generate some random points to compute distances
    Sigma = rng.random(size=2, dtype=dtype)
    Sigma @= Sigma.T
    x = rng.multivariate_normal(np.zeros(2, dtype=dtype), Sigma, size=mat.shape[0])

    # compute square distances
    D = x.reshape(-1, 1, 2) - x.reshape(1, -1, 2)
    D = np.sum(D * D, axis=2)

    # compute Gaussian distances
    D = np.exp(-D / (2.0 * sigma**2))

    return mat * D  # type: ignore[no-any-return]


def generate_random_equal_row_sum(
    mat: Array,
    *,
    alpha: float = 1.0,
    dtype: Any = None,
    rng: np.random.Generator | None = None,
) -> Array:
    r"""Generate a random weights for the adjacency matrix *mat* with equal row sum.

    This is roughly equivalent to generating a random matrix using
    :func:`generate_random_weights` and then normalizing using
    :func:`normalize_equal_row_sum`. However, it uses the Dirichlet distribution
    to generate more theoretically sound random numbers.

    :arg alpha: parameter in the Dirichlet distribution. A value :math:`\alpha > 1`
        will result in values that approach the uniform :math:`1 / k` weight,
        where :math:`k` is the number of connections on the row. A value of 1
        gives a standard uniform Dirichlet distribution.
    """

    if rng is None:
        rng = np.random.default_rng()

    n = mat.shape[0]
    result = np.zeros(mat.shape, dtype=dtype)
    for i in range(n):
        (j,) = np.where(mat[i] == 1)
        if j.size == 0:
            continue

        weights = rng.dirichlet([alpha] * j.size)
        result[i, j] = weights

    return result


def generate_symmetric_random_equal_row_sum(
    mat: Array,
    *,
    maxit: int = 512,
    atol: float = 1.0e-9,
    dtype: Any = None,
    rng: np.random.Generator | None = None,
) -> Array:
    """This generates a symmetric random matrix with equal row sum of 1.

    By definition, a symmetric matrix with equal row sum also has equal column
    sum. Such a matrix is generated using the Sinkhorn-Knopp algorithm. Note that
    the row and column sums will only be equal to 1 to the given tolerance
    *atol*.

    :arg atol: absolute tolerance for the row and column sums.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Adjacency matrix should be square: {mat.shape}")

    if maxit <= 0:
        raise ValueError(f"'maxit' should be positive: {maxit}")

    if atol <= 0:
        raise ValueError(f"'rtol' should be positive: {atol}")

    if rng is None:
        rng = np.random.default_rng()

    # generate an initial symmetric random matrix
    result = rng.uniform(0.01, 1.0, size=mat.shape).astype(dtype)
    result += result.T

    # scaling vector
    r = c = np.ones(mat.shape[0], dtype=dtype)

    i = 0
    converged = False

    while i < maxit and not converged:
        cprev = c

        r = 1.0 / (result @ c)
        c = 1.0 / (result.T @ r)

        # NOTE: Symmetric Sinkhorn--Knopp algorithm could oscillate between two
        # sequences, need to bring the two sequences together (See for example
        # "Algorithms For The Equilibration Of Matrices And Their Application
        # To Limited-memory Quasi-newton Methods")
        c = (c + r) / 2.0

        converged = bool(la.norm(c - cprev, ord=2) < atol)
        i += 1

    d = np.diag(r)
    return d @ result @ d  # type: ignore[no-any-return]


def normalize_equal_row_sum(
    mat: Array,
    *,
    diagonal: bool = False,
) -> Array:
    r"""Take a weight matrix *mat* and ensure it has equal row sum of *1*.

    :arg diagonal: if *True*, only the diagonal of *mat* is updated so that the
        result has equal row sum. Otherwise, the whole row is scaled.
    """

    # NOTE: all our adjacency matrices have elements on all rows and the weights
    # are generated on [0.01, 1], so there is no reason a row would have zero sum
    fac = np.sum(mat, axis=1)
    if np.any(np.abs(fac) < 10 * np.finfo(mat.dtype).eps):
        raise ValueError("Matrix has a zero sum row")

    if diagonal:
        result = mat.copy()
        np.fill_diagonal(result, 1.0 - fac + np.diag(mat))
    else:
        result = mat / fac.reshape(-1, 1)

    return result  # type: ignore[no-any-return]


# }}}
