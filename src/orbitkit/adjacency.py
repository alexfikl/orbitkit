# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.linalg as la

from orbitkit.typing import Array1D, Array2D
from orbitkit.utils import module_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike

log = module_logger(__name__)


# {{{ utils


def compute_graph_density(mat: Array2D[np.floating[Any]]) -> float:
    """Compute the density of the adjacency matrix *mat*.

    The density is defined as the number of edges in the graph divided by the
    maximum possible number of edges for the given node count. It is always a
    number in :math:`[0, 1]`.

    :arg mat: a binary adjacency matrix.
    """
    if mat.ndim != 2:
        raise ValueError(f"adjacency matrix is not 2 dimensional: {mat.shape}")

    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"adjacency matrix is not square: {mat.shape}")

    n, _ = mat.shape
    if n == 1:
        return 0.0

    # NOTE: this subtracts the diagonal so that we can handle graphs with self-loops
    edges = np.sum(mat) - np.sum(np.diag(mat))
    max_edges = n * (n - 1)

    return float(edges / max_edges)


def compute_graph_triangles(mat: Array2D[np.floating[Any]]) -> int:
    r"""Compute number of triangles in the graph with adjacency matrix *mat*.

    The number of triangles in a graph is given by the simple formula

    .. math::

        \frac{\text{trace}(A^3)}{6}

    :arg mat: a binary adjacency matrix.
    """
    if mat.ndim != 2:
        raise ValueError(f"adjacency matrix is not 2 dimensional: {mat.shape}")

    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"adjacency matrix is not square: {mat.shape}")

    n, _ = mat.shape
    if n <= 2:
        return 0

    # NOTE: this computes something like
    #   tr(O^3) = tr(A^3) - 3 * sum A_{ii} * (A^2)_{ii} + 2 sum A_{ii}^3
    #   O = A - D
    # so that we can handle matrices with self-loops as well.
    d = np.diag(mat)
    mat2 = mat @ mat
    trmat3 = np.trace(mat2 @ mat)
    trmat3 = trmat3 - 3 * d @ np.diag(mat2) + 2 * np.sum(d**3)

    return int(trmat3) // 6


def make_graph_laplacian_undirected(
    A: Array2D[np.floating[Any]], *, normalize: bool = False
) -> Array2D[np.floating[Any]]:
    r"""Compute the graph Laplacian for the adjacency matrix *A*.

    .. math::

        L = D - A \implies L_{ij} =
        \begin{cases}
        \text{deg}(v_i), & \quad \text{if } i = j, \\
        -1, & \quad \text{if } i \ne j \text{ and } A_{ij} = 1, \\
        0, & \quad \text{otherwise},
        \end{cases}

    where the degree is the number of vertices connected to :math:`v_i`, including
    self-loops. Note that *A* is assumed to be symmetric for an undirected graph.
    For the normalization, we use

    .. math::

        L_{\text{norm}} = D^{-\frac{1}{2}} L D^{-\frac{1}{2}}.
    """

    assert np.allclose(A, A.T)

    D = np.sum(A, axis=1)
    L = -A
    np.fill_diagonal(L, D)
    if normalize:
        Dinv = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
        L = Dinv[:, None] * L * Dinv[None, :]

    assert L.shape == A.shape
    return L


def make_graph_laplacian_directed(
    A: Array2D[np.floating[Any]],
    *,
    out: bool = True,
    normalize: bool = False,
) -> Array2D[np.floating[Any]]:
    r"""Compute the graph Laplacian for the adjacency matrix *A*.

    For the normalization, we use left or right normalization, depending on the
    value of *out*. We take

    .. math ::

        L_{\text{norm}} =
            \begin{cases}
            D_{\text{out}}^{-1} L_{\text{out}}, \\
            L_{\text{in}} D_{\text{in}}^{-1}.
            \end{cases}

    :arg out: if *True*, we compute the out-degree Laplacian. Otherwise, we
        compute the in-degree Laplacian.
    """

    L = -A
    if out:
        D = np.sum(A, axis=1)
        np.fill_diagonal(L, D)

        if normalize:
            Dinv = np.where(D > 0, 1.0 / D, 0.0)
            L = Dinv[:, None] * L
    else:
        D = np.sum(A, axis=0)
        np.fill_diagonal(L, D)

        if normalize:
            Dinv = np.where(D > 0, 1.0 / D, 0.0)
            L = L * Dinv[None, :]  # noqa: PLR6104

    assert L.shape == A.shape
    return L


def stringify_adjacency(mat: Array2D[np.floating[Any]], *, fmt: str = "box") -> str:
    """Stringify a (preferably binary) adjacency matrix.

    The supported formats are:

    * ``box``: each entry is a box (3 chars per entry). This works best for smaller
      graphs (depending on your display width).
    * ``tight``: only connected entries are a box (2 chars per entry). This
      works best for medium graphs (again depending on your display width).
    * ``braille``: an even tighter representation, where each :math:`4 \times 2`
      block is averaged and encoded into a little braille character. This should
      work best for pretty large graphs.
    * ``latex``: a ``bmatrix`` environment with the matrix entries, written out
      verbatim.
    """
    if fmt == "box":
        symbols = {0: " ◻ ", 1: " ◼ "}

        return "\n".join(
            "".join(symbols[int(mat[i, j] != 0)] for j in range(mat.shape[1]))
            for i in range(mat.shape[0])
        )
    elif fmt == "tight":
        symbols = {0: "  ", 1: "▒▒"}

        return "\n".join(
            "".join(symbols[int(mat[i, j] != 0)] for j in range(mat.shape[1]))
            for i in range(mat.shape[0])
        )
    elif fmt == "braille":
        height = (mat.shape[0] // 4) * 4
        width = (mat.shape[1] // 2) * 2
        mat = (mat[:height, :width] != 0).astype(np.uint8)

        result = []
        for i in range(0, height, 4):
            line = []
            for j in range(0, width, 2):
                dots = int(
                    mat[i + 0, j + 0] << 0
                    | mat[i + 1, j + 0] << 1
                    | mat[i + 2, j + 0] << 2
                    | mat[i + 0, j + 1] << 3
                    | mat[i + 1, j + 1] << 4
                    | mat[i + 2, j + 1] << 5
                    | mat[i + 3, j + 0] << 6
                    | mat[i + 3, j + 1] << 7
                )

                line.append(chr(0x2800 + dots))

            result.append("".join(line))

        return "\n".join(result)
    elif fmt == "latex":
        lines = []

        lines.append(r"\begin{bmatrix}")
        for i in range(mat.shape[0]):
            lines.append(" & ".join(str(mij) for mij in mat[i]))
            lines.append(r"\\")
        lines.append(r"\end{bmatrix}")

        return "\n".join(lines)
    else:
        raise ValueError(f"unknown stringify format: '{fmt}'")


ADJACENCY_TYPES = frozenset({
    "bus",
    "bus1",
    "bus2",
    "configuration",
    "erdosrenyi",
    "feedforward",
    "fractal",
    # "gapjunctions",
    "lattice",
    "ring",
    "ring1",
    "ring2",
    "star",
    "startree",
    "strogatzwatts",
    "barabasialbert",
    "barabasialbert2",
    "barabasialbert4",
    "distancedecay",
})


def make_adjacency_matrix_from_name(  # noqa: PLR0911
    n: int,
    topology: str,
    *,
    k: int | None = None,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    """
    :arg k: number of neighbor connections in a ring, bus, or Strogatz-Watts
        network; average number of neighbors in a Erdős-Rényi network; number of
        gap junction clusters; number of levels in a fractal network.
    """
    if dtype is None:
        dtype = np.int32

    if rng is None:
        rng = np.random.default_rng()

    if topology == "feedforward":
        return generate_adjacency_feed_forward(n, dtype=dtype)
    elif topology == "ring":
        k = 2 if k is None else k
        return generate_adjacency_ring(n, k=k, dtype=dtype)
    elif topology == "ring1":
        return generate_adjacency_ring(n, k=1, dtype=dtype)
    elif topology == "ring2":
        return generate_adjacency_ring(n, k=2, dtype=dtype)
    elif topology == "bus":
        k = 2 if k is None else k
        return generate_adjacency_bus(n, k=k, dtype=dtype)
    elif topology == "bus1":
        return generate_adjacency_bus(n, k=1, dtype=dtype)
    elif topology == "bus2":
        return generate_adjacency_bus(n, k=2, dtype=dtype)
    elif topology == "star":
        return generate_adjacency_star(n, dtype=dtype)
    elif topology == "startree":
        return generate_adjacency_star_tree(n, dtype=dtype)
    elif topology == "lattice":
        return generate_adjacency_lattice(n, dtype=dtype)
    elif topology == "erdosrenyi":
        return generate_adjacency_erdos_renyi(n, k=k, dtype=dtype, rng=rng)
    elif topology == "strogatzwatts":
        k = 2 if k is None else k
        return generate_adjacency_strogatz_watts(n, k=k, dtype=dtype, rng=rng)
    elif topology == "barabasialbert":
        k = 2 if k is None else k
        return generate_adjacency_barabasi_albert(n, k, dtype=dtype, rng=rng)
    elif topology == "barabasialbert2":
        return generate_adjacency_barabasi_albert(n, 2, dtype=dtype, rng=rng)
    elif topology == "barabasialbert4":
        return generate_adjacency_barabasi_albert(n, 4, dtype=dtype, rng=rng)
    elif topology == "distancedecay":
        return generate_adjacency_distance_decay(n, dtype=dtype, rng=rng)
    elif topology == "configuration":
        return generate_adjacency_configuration(n, dtype=dtype, rng=rng)
    elif topology == "gapjunctions":
        k = max(int(n // 9) - 1, 1) if k is None else k
        return generate_adjacency_gap_junctions(n, k, dtype=dtype, rng=rng)
    elif topology == "fractal":
        # NOTE: for k levels, we have n = p**{k + 1} nodes in the network
        k = 4 if k is None else k
        p = round(n ** (1 / (k + 1)))
        base = "".join(f"{rng.integers(2)}" for _ in range(p))
        return generate_adjacency_fractal(base, nlevels=k, dtype=dtype)
    else:
        raise ValueError(f"unknown topology: '{topology}'")


# }}}

# {{{ adjacency matrices


def generate_adjacency_all(
    n: int, *, dtype: DTypeLike | None = None
) -> Array2D[np.floating[Any]]:
    r"""Generate a all-to-all :math:`n \times n` adjacency matrix."""
    if dtype is None:
        dtype = np.int32

    result = np.ones((n, n), dtype=dtype)
    np.fill_diagonal(result, 0)

    return result


def generate_adjacency_feed_forward(
    n: int, *, dtype: DTypeLike | None = None
) -> Array2D[np.floating[Any]]:
    r"""Generate a :math:`n \times n` lower triangular adjacency matrix."""
    if dtype is None:
        dtype = np.int32

    result = np.ones((n, n), dtype=dtype)
    return np.tril(result, k=-1)


def generate_adjacency_ring(
    n: int, *, k: int = 1, dtype: DTypeLike | None = None
) -> Array2D[np.floating[Any]]:
    """Generate a *k*-ring network with :math:`n` nodes.

    In this network, each node is connected to its :math:`k` nearest neighbors
    with periodicity. For a non-periodic version see :func:`generate_adjacency_bus`.
    """
    if dtype is None:
        dtype = np.int32

    if n < 0:
        raise ValueError(f"negative dimensions are now allowed: '{n}'")

    if n == 1:
        return np.zeros((n, n), dtype=dtype)

    if not 0 <= k < n:
        raise ValueError(f"number of neighbors 'm' is invalid: '{k}' (not in [0, {n})")

    # NOTE: this is essentially just a periodic banded matrix
    eye = np.eye(n, dtype=dtype)
    result = np.zeros((n, n), dtype=dtype)

    for i in range(-k, k + 1):
        if i == 0:
            continue

        result += np.roll(eye, i, axis=1)

    return result


def generate_adjacency_bus(
    n: int, *, k: int = 1, dtype: DTypeLike | None = None
) -> Array2D[np.floating[Any]]:
    """Generate a bus network with :math:`n` nodes.

    In this network, each node is connected to its :math:`k` nearest neighbors
    in a non-periodic fashion.
    """
    if dtype is None:
        dtype = np.int32

    if n < 0:
        raise ValueError(f"negative dimensions are now allowed: '{n}'")

    if n == 1:
        return np.zeros((n, n), dtype=dtype)

    if not 0 <= k < n:
        raise ValueError(f"number of neighbors 'm' is invalid: '{k}' (not in [0, {n}])")

    # NOTE: this is essentially just a non-periodic banded matrix
    ones = np.ones(n, dtype=dtype)
    result = np.zeros((n, n), dtype=dtype)

    for i in range(-k, k + 1):
        if i == 0:
            continue

        result += np.diag(ones[abs(i) :], k=i)

    return result


def generate_adjacency_star(
    n: int, *, dtype: DTypeLike | None = None
) -> Array2D[np.floating[Any]]:
    """Generate a star network with :math:`n` nodes.

    In this network, there is a central node connected to all nodes.
    """
    if dtype is None:
        dtype = np.int32

    if n < 0:
        raise ValueError(f"negative dimensions are now allowed: '{n}'")

    result = np.zeros((n, n), dtype=dtype)
    result[0, 1:] = 1
    result[1:, 0] = 1

    return result


def generate_adjacency_star_tree(
    n: int,
    *,
    nhubs: int | None = None,
    dtype: DTypeLike | None = None,
) -> Array2D[np.floating[Any]]:
    """Generate a star of stars network with :math:`n` nodes.

    In this setup, the network will have:
    * a central hub node.
    * *nhubs* nodes connected to the central node.
    * the remaining nodes will be equally distributed across the sub-hub nodes.

    :arg nhubs: number of hubs connected to the central hub node. By default,
        this depends on *n*.
    """

    if nhubs is None:
        nhubs = max(n // 5, 1)

    if dtype is None:
        dtype = np.int32

    if n < 0:
        raise ValueError(f"negative dimensions are now allowed: '{n}'")

    if n == 0 or nhubs == 1:
        return generate_adjacency_star(n, dtype=dtype)

    if nhubs < 0:
        raise ValueError(f"negative number of hubs is now allowed: '{nhubs}'")

    if nhubs == 0:
        raise ValueError("zero hubs are not allowed")

    if n < nhubs + 1:
        raise ValueError(
            f"number of nodes must be higher than number of hubs: {n} < {nhubs}"
        )

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

    return result


def _find_equal_factors(n: int) -> tuple[int, int]:
    if n == 0:
        return 0, 0

    m = int(np.sqrt(n)) + 1
    while n % m != 0:
        m -= 1

    assert m > 0
    return n // m, m


def generate_adjacency_lattice(
    n: int,
    m: int | None = None,
    *,
    dtype: DTypeLike | None = None,
) -> Array2D[np.floating[Any]]:
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

    if n < 0:
        raise ValueError(f"negative dimensions are now allowed: '{n}'")

    if m is None:
        n, m = _find_equal_factors(n)

    if m < 0:
        raise ValueError(f"'m' cannot be non-positive: '{m}'")

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
    dtype: DTypeLike | None = None,
    symmetric: bool = True,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    r"""Generate a random Erdős-Rényi :math:`n \times n` adjacency matrix.

    :arg p: probability of an edge between two nodes (defaults to *0.25*).
    :arg k: average number of edges for each node (i.e. the degree). If *p* is
        not given, it is computed as :math:`p = k / (n - 1)`.
    :arg symmetric: if *True*, the adjacency matrix will be symmetric.
    """
    if n < 0:
        raise ValueError(f"negative dimensions are now allowed: '{n}'")

    if p is not None and k is not None:
        raise ValueError("cannot pass both 'p' and 'k'")

    if p is not None and not 0.0 <= p <= 1.0:
        raise ValueError(f"probability 'p' not in [0, 1]: '{p}'")

    if k is not None and not 0 <= k < n:
        raise ValueError(f"cannot have more edges 'k' than nodes 'n': {k} > {n}")

    if p is None:
        p = 0.25 if k is None else (k / (n - 1))

    if dtype is None:
        dtype = np.int32

    if rng is None:
        rng = np.random.default_rng()

    if symmetric:
        rows, cols = np.tril_indices(n, k=-1)
        mask = rng.random(size=rows.size) < p

        result = np.zeros((n, n), dtype=dtype)
        result[rows[mask], cols[mask]] = 1
        result[cols[mask], rows[mask]] = 1
    else:
        # NOTE: setting the diagonal to zero should not change the statistics
        result = (rng.random(size=(n, n)) < p).astype(dtype)
        np.fill_diagonal(result, 0)

    return result


def generate_adjacency_strogatz_watts(
    n: int,
    *,
    k: int = 2,
    p: float = 0.1,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    r"""Generate a random Strogatz-Watts :math:`n \times n` adjacency matrix.

    :arg k: number of neighboring nodes.
    :arg p: rewiring probability.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability 'p' not in [0, 1]: '{p}'")

    if dtype is None:
        dtype = np.int32

    if rng is None:
        rng = np.random.default_rng()

    result = generate_adjacency_ring(n, k=k, dtype=dtype)
    for i in range(n):
        forbidden = {(i + j) % n for j in range(-k, k + 1)}
        choices = np.array([c for c in range(n) if c not in forbidden])

        for j in range(1, k + 1):
            if not rng.random() < p:
                continue

            # remove current edge
            jold = (i + j) % n
            result[i, jold] = result[jold, i] = 0

            # rewire to a new edge
            jnew = rng.choice(choices)
            result[i, jnew] = result[jnew, i] = 1

    return result


def generate_adjacency_barabasi_albert(
    n: int,
    m: int,
    *,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    """Generate a random Barabási-Albert adjacency matrix.

    :arg m: number of edges each new node should attach to :math:`m < n`.
    """
    if n < 0:
        raise ValueError(f"negative dimensions are now allowed: '{n}'")

    if m < 0:
        raise ValueError(f"negative number of edges is now allowed: '{m}'")

    if m >= n:
        raise ValueError(f"invalid sizes (m >= n): {m} >= {n}")

    if dtype is None:
        dtype = np.int32

    if rng is None:
        rng = np.random.default_rng()

    mat = np.zeros((n, n), dtype=dtype)

    # construct an initial dense block of m + 1 nodes with m edges
    mat[: m + 1, : m + 1] = 1
    np.fill_diagonal(mat, 0)

    # iteratively add more nodes
    for i in range(m + 1, n):
        # construct probability based on degree: larger degree nodes get a larger
        # probability of additional connections => even bigger group!
        d = np.sum(mat[:i, :i], axis=1)
        p = d / np.sum(d)

        # randomly choose some connections
        j = rng.choice(i, size=m, replace=False, p=p)
        mat[i, j] = 1
        mat[j, i] = 1

    return mat


def generate_adjacency_distance_decay(
    n: int,
    *,
    xlim: tuple[float, float] = (0, 1),
    ylim: tuple[float, float] | None = None,
    beta: float = 1.0,
    sigma: float | None = None,
    symmetric: bool = False,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    r"""Construct a random adjacency matrix based on distance.

    This function generates (uniform) random points in the given limits *xlim*
    and *ylim*. From these, we construct probabilities

    .. math ::

        p_{ij} = \beta \exp\left(-\frac{\|x_i - x_j\|}{\sigma}\right)

    These are then used to determine the structure of the adjacency matrix by
    additional random sampling. If *symmetric* is *true*, the matrix is forcibly
    symmetrized.

    :arg xlim: limits for the x coordinate.
    :arg ylim: limits for the y coordinate (defaults to the x coordinate limits).
    :arg beta: a probability scaling that must be in :math:`(0, 1]`.
    :arg sigma: variance-like parameter that controls how probable connections to
        farther away vertices is. Defaults to 0.1 of the domain size.
    """
    if n < 0:
        raise ValueError(f"negative dimensions are now allowed: '{n}'")

    if not 0.0 < beta <= 1.0:
        raise ValueError(f"'beta' must be in (0, 1]: {beta}")

    if xlim[0] > xlim[1]:
        xlim = (xlim[1], xlim[0])

    if ylim is None:
        ylim = xlim

    if ylim[0] > ylim[1]:
        ylim = (ylim[1], ylim[0])

    if sigma is None:
        sigma = 0.2 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])

    if sigma <= 0.0:
        raise ValueError(f"'sigma' cannot be negative: {sigma}")

    if dtype is None:
        dtype = np.int32

    if rng is None:
        rng = np.random.default_rng()

    # generate some random points
    x = np.stack([
        rng.uniform(xlim[0], xlim[1], size=n),
        rng.uniform(ylim[0], ylim[1], size=n),
    ])
    d = np.sqrt(np.sum((x[:, None, :] - x[:, :, None]) ** 2, axis=0))

    # probabilities
    p = beta * np.exp(-d / sigma)

    # adjacency
    mat = rng.random((n, n))
    mat = (mat < p).astype(dtype)

    np.fill_diagonal(mat, 0)
    if symmetric:
        tril = np.tril_indices(n, k=-1)
        mat[tril] = mat.T[tril]

    return mat


def _generate_random_gap_junction_clusters(
    rng: np.random.Generator,
    n: int,
    m: int,
    *,
    alpha: float,
    avgsize: int,
    maxsize: int,
    maxiter: int,
) -> Array2D[np.floating[Any]]:
    x = np.array([n // m] * m, dtype=np.int64)

    # FIXME: this seems like it'll have mean *mean* only if n > mean * m?
    smax = min(n - 1, avgsize * m)
    for _ in range(maxiter):
        # generate candidates
        p = rng.dirichlet((alpha,) * m)
        x = np.rint(p * smax).astype(x.dtype)

        # ensure they sum up to smax
        extra = smax - np.sum(x)
        if extra != 0:
            idx = rng.choice(m, size=abs(extra), replace=True)
            x[idx] += np.sign(extra)

        # check that the maximum size is respected
        if np.max(x) <= maxsize and np.min(x) >= 1:
            break

    return x


def _make_adjacency_from_groups(
    groups: Array1D[np.integer[Any]],
    gaps: int | Array1D[np.integer[Any]],
    *,
    dtype: DTypeLike | None = None,
) -> tuple[
    Array2D[np.floating[Any]], Array1D[np.integer[Any]], Array1D[np.integer[Any]]
]:
    if dtype is None:
        dtype = np.int32

    if isinstance(gaps, int):
        gaps = np.array([gaps] * groups.size)

    n = int(np.sum(groups) + np.sum(gaps))
    if groups.shape != gaps.shape:
        raise ValueError(
            "cluster sizes and gap sizes must have the same shape: "
            f"got {groups.shape} and {gaps.shape}"
        )

    i = 0
    result = np.zeros((n, n), dtype=dtype)

    for m, g in zip(groups, gaps, strict=True):
        result[i : i + m, i : i + m] = 1.0
        i += m + g

    np.fill_diagonal(result, 0)
    return result, groups, gaps


def generate_adjacency_gap_junctions(
    n: int,
    m: int,
    *,
    dtype: DTypeLike | None = None,
    alpha: float = 1.0,
    avgsize: int = 9,
    maxsize: int = 21,
    maxiter: int = 512,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    r"""Generate an adjacency matrix for gap junctions in a neuron network.

    A neuron network with gap junctions is generally represented as a set of
    unconnected all-to-all subnetworks. The defaults in this function are chosen
    for the TRN (Thalamic Reticular Nucleus) based on the work from [Lee2014]_.
    Note that other brain regions have very different distributions, so better
    values should be chosen if a realistic application is desired.

    .. [Lee2014] S.-C. Lee, S. L. Patrick, K. A. Richardson, B. W. Connors,
        *Two Functionally Distinct Networks of Gap Junction-Coupled Inhibitory
        Neurons in the Thalamic Reticular Nucleus*,
        The Journal of Neuroscience, Vol. 34, pp. 13170--13182, 2014,
        `doi:10.1523/jneurosci.0562-14.2014 <https://doi.org/10.1523/jneurosci.0562-14.2014>`__.

    :arg n: the number of nodes in the network.
    :arg m: the desired number of gap junction clusters. This should be such that
        :math:`n > m \times \text{avgsize}` to allow clusters of the desired size
        distribution. If this is not the case, the average cluster size of the
        generated network will be smaller.
    :arg alpha: parameter in the Dirichlet distribution used to generate gap
        junction clusters.
    :arg avgsize: desired mean for the gap junction cluster size. Note that this
        function always generates clusters with mean exactly *avgsize*.
    :arg maxsize: maximum size of a gap junction cluster. Note that it is not
        guaranteed that a cluster with this maximum size will exist in the network.
    :arg maxiter: the gap junction clusters are generated by an iterative algorithm.
        This defines the maximum number of iterations that can be used.
    """

    if n < 0:
        raise ValueError(f"negative dimensions are now allowed: '{n}'")

    if m < 0:
        raise ValueError(f"negative cluster counts are now allowed: '{m}'")

    if dtype is None:
        dtype = np.int32

    if rng is None:
        rng = np.random.default_rng()

    if avgsize > maxsize:
        raise ValueError(
            f"'avgsize' cannot be larger than 'maxsize': {avgsize} > {maxsize}"
        )

    if m == 0 or n <= 1:
        return np.zeros((n, n), dtype=dtype)

    # generate gap junction clusters
    groups = _generate_random_gap_junction_clusters(
        rng,
        n,
        m,
        alpha=alpha,
        avgsize=avgsize,
        maxsize=maxsize,
        maxiter=maxiter,
    )
    leftover = n - np.sum(groups)

    # generate random gaps
    cuts = rng.choice(np.arange(1, leftover), size=m - 1, replace=False)
    pts = np.concatenate(([0], np.sort(cuts), [leftover]))
    gaps = np.diff(pts)
    assert (np.sum(groups) + np.sum(gaps)) == n

    # create adjacency matrix
    result, _, _ = _make_adjacency_from_groups(groups, gaps, dtype=dtype)
    assert result.shape == (n, n)

    return result


def _expand_pattern(
    base: str, nlevels: int, dtype: DTypeLike | None = None
) -> Array2D[np.floating[Any]]:
    zeros = "0" * len(base)
    pattern = base
    for _ in range(nlevels - 1):
        pattern = "".join([(base if i == "1" else zeros) for i in pattern])

    # transform pattern to 0/1 array
    return np.fromiter(pattern, dtype=dtype)


def generate_adjacency_fractal(
    base: str,
    *,
    nlevels: int = 4,
    dtype: DTypeLike | None = None,
) -> Array2D[np.floating[Any]]:
    """Generate a Cantor set-like connectivity based on the *base* pattern.

    This function generates the network described in [Omelchenko2015]_. It takes
    a base pattern (e.g. ``"11011"``) and performs a recursive Cantor set
    construction, where each "1" is replaced by the base pattern and each "0"
    is replaced by a zero pattern the same length as the base pattern. This
    results in a networks of size ``len(base) ** nlevels`` after all the
    subdivisions.

    The resulting pattern is then used to construct a circulant adjacency
    matrix that can be conveniently arranged in non-standard ring-structure.

    .. [Omelchenko2015] I. Omelchenko, A. Provata, J. Hizanidis, E. Schöll, P. Hövel,
        *Robustness of Chimera States for Coupled FitzHugh-Nagumo Oscillators*,
        Physical Review E, Vol. 91, pp. 22917--22917, 2015,
        `doi:10.1103/physreve.91.022917 <https://doi.org/10.1103/physreve.91.022917>`__.

    :arg nlevels: number of levels the pattern is subdivided.
    """

    if not set(base) <= {"0", "1"}:
        raise ValueError(f"'base' pattern must be binary only: '{base}'")

    if nlevels < 0:
        raise ValueError(f"'nlevels' must be non-negative: {nlevels}")

    if dtype is None:
        dtype = np.int32

    if nlevels == 0:
        # FIXME: what size do we actually expect here? this chose was mostly
        # made to fit match `len(base) ** 0`
        return np.zeros((1, 1), dtype=dtype)

    from scipy.linalg import circulant

    x = _expand_pattern(base, nlevels, dtype=dtype)
    mat = circulant(x).T
    np.fill_diagonal(mat, 0)

    return mat


def generate_adjacency_configuration(
    n: int,
    *,
    degrees: Array1D[np.integer[Any]] | int | None = None,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    """Generate a random :math:`n \times n` adjacency matrix for a configuration
    model.

    :arg degrees: an array of desired node degrees that should sum up to an even
        number.
    """

    if dtype is None:
        dtype = np.int32

    if rng is None:
        rng = np.random.default_rng()

    if degrees is None:
        degrees = 10

    if isinstance(degrees, int):
        while True:
            result = rng.integers(1, degrees, size=n, dtype=dtype)
            if np.sum(result) % 2 == 0:
                break

        degrees = result

    if np.sum(degrees) % 2 != 0:
        raise ValueError(f"degrees should sum up to an even number: {degrees}")

    # create pairwise stubs
    stubs = np.repeat(np.arange(n), degrees)
    rng.shuffle(stubs)

    # create adjacency matrix
    from itertools import islice

    # TODO: replace with itertools.batched once we depend on Python >= 3.12
    def batched(iterable, n, *, strict=False):
        if n < 1:
            raise ValueError("n must be at least one")

        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            if strict and len(batch) != n:
                raise ValueError("batched(): incomplete batch")
            yield batch

    result = np.zeros((n, n), dtype=dtype)
    for i, j in batched(stubs, n=2):
        if i == j:
            continue

        result[i, j] = result[j, i] = 1

    return result


# }}}

# {{{ weights


def generate_random_weights(
    mat: Array2D[np.floating[Any]],
    *,
    dtype: DTypeLike | None = None,
    symmetric: bool = False,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
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

    return mat * w


def generate_random_gaussian_weights(
    mat: Array2D[np.floating[Any]],
    *,
    dtype: DTypeLike | None = None,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    """Generate a random weight matrix based on Gaussian node distance.

    :arg sigma: standard deviation of the normal distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    # generate some random points to compute distances
    Sigma = rng.random(size=2, dtype=dtype)  # ty: ignore[no-matching-overload]
    Sigma @= Sigma.T
    x = rng.multivariate_normal(np.zeros(2, dtype=dtype), Sigma, size=mat.shape[0])

    # compute square distances
    D = x.reshape(-1, 1, 2) - x.reshape(1, -1, 2)
    D = np.sum(D * D, axis=2)

    # compute Gaussian distances
    D = np.exp(-D / (2.0 * sigma**2))

    return mat * D


def generate_random_equal_row_sum(
    mat: Array2D[np.floating[Any]],
    *,
    alpha: float = 1.0,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
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
    mat: Array2D[np.floating[Any]],
    *,
    maxit: int = 512,
    atol: float = 1.0e-9,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    """This generates a symmetric random matrix with equal row sum of 1.

    By definition, a symmetric matrix with equal row sum also has equal column
    sum. Such a matrix is generated using the Sinkhorn-Knopp algorithm. Note that
    the row and column sums will only be equal to 1 to the given tolerance
    *atol*.

    :arg atol: absolute tolerance for the row and column sums.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"adjacency matrix should be square: {mat.shape}")

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
        # To Limited-memory Quasi-Newton Methods")
        c = (c + r) / 2.0

        converged = bool(la.norm(c - cprev, ord=2) < atol)
        i += 1

    d = np.diag(r)
    return d @ result @ d


def apply_graph_laplacian(
    mat: Array2D[np.floating[Any]],
    f: Callable[[Array1D[np.inexact[Any]]], Array1D[np.inexact[Any]]],
    *,
    out: bool | None = None,
) -> Array2D[np.floating[Any]]:
    r"""Apply the function *f* to the graph Laplacian of the adjacency matrix *mat*.

    We set the weights to be :math:`\boldsymbol{W} = f(\boldsymbol{L})`. Applying
    the function to the graph Laplacian is done spectrally, i.e. given the eigen
    decomposition, we write

    .. math ::

        \boldsymbol{W} = \boldsymbol{U} f(\boldsymbol{\Lambda}) \boldsymbol{U}^{-1}.

    Note that, if the adjacency matrix is not symmetric, this may result in
    a complex weight matrix.
    """

    if out is None:
        L = make_graph_laplacian_undirected(mat)
        sigma, U = np.linalg.eigh(L)
        Uinv = U.T
    else:
        L = make_graph_laplacian_directed(mat, out=out)
        sigma, U = np.linalg.eig(L)
        Uinv = np.linalg.inv(U)

    return U @ np.diag(f(sigma)) @ Uinv


def generate_graph_laplacian_weights(
    mat: Array2D[np.floating[Any]],
    f: Callable[[Array1D[np.inexact[Any]]], Array1D[np.inexact[Any]]],
    *,
    out: bool | None = None,
) -> Array2D[np.floating[Any]]:
    r"""Generate weights based on the graph Laplacian of *mat*.

    This functionu uses :func:`apply_graph_laplacian` to get a set of weights.
    It then ensures that diagonal is zero and the weights are positive.
    """

    W = apply_graph_laplacian(mat, f, out=out)
    np.fill_diagonal(W, 0.0)

    return W


def normalize_equal_row_sum(
    mat: Array2D[np.floating[Any]],
    *,
    diagonal: bool = False,
) -> Array2D[np.floating[Any]]:
    r"""Take a weight matrix *mat* and ensure it has equal row sum of *1*.

    :arg diagonal: if *True*, only the diagonal of *mat* is updated so that the
        result has equal row sum. Otherwise, the whole row is scaled.
    """

    # NOTE: all our adjacency matrices have elements on all rows and the weights
    # are generated on [0.01, 1], so there is no reason a row would have zero sum
    fac = np.sum(mat, axis=1)
    if np.any(np.abs(fac) < 10 * np.finfo(mat.dtype).eps):
        raise ValueError("matrix has a zero sum row")

    if diagonal:
        result = mat.copy()
        np.fill_diagonal(result, 1.0 - fac + np.diag(mat))
    else:
        result = mat / fac.reshape(-1, 1)

    return result


# }}}
