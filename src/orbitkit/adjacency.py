# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

import array_api_compat
import numpy as np
import numpy.linalg as la

from orbitkit.array_api_extras import AArray1D, AArray2D, DTypeLike
from orbitkit.utils import module_logger

if TYPE_CHECKING:
    from collections.abc import Callable

log = module_logger(__name__)


# {{{ utils


def stringify_adjacency(
    mat: AArray2D[np.floating[Any]],
    *,
    fmt: str | None = None,
) -> str:
    r"""Stringify a (preferably binary) adjacency matrix.

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
    if not array_api_compat.is_numpy(mat):
        raise TypeError(f"'mat' must be a 'numpy' array: {type(mat)}")

    if fmt is None:
        n = max(mat.shape)
        if n < 50:
            fmt = "box"
        elif n < 75:
            fmt = "tight"
        else:
            fmt = "braille"

    if fmt == "box":
        symbols = {0: " ◻ ", 1: " ◼ "}

        return "\n".join(
            "".join(symbols[int(mat[i, j] != 0)] for j in range(mat.shape[1]))
            for i in range(mat.shape[0])
        )
    elif fmt == "tight":
        symbols = {0: "··", 1: "▒▒"}

        return "\n".join(
            "".join(symbols[int(mat[i, j] != 0)] for j in range(mat.shape[1]))
            for i in range(mat.shape[0])
        )
    elif fmt == "braille":
        height = (mat.shape[0] // 4) * 4
        width = (mat.shape[1] // 4) * 4
        mat = np.astype(mat[:height, :width] != 0, np.uint8)

        def braille(block: np.ndarray[tuple[int, int], np.dtype[np.uint8]]) -> str:
            offset = sum(int(b) << i for i, b in enumerate(block.flat))
            return " " if offset == 0 else chr(0x2800 + offset)

        result = []
        for i in range(0, height, 4):
            line = []
            for j in range(0, width, 4):
                line.append(
                    braille(mat[i : i + 4, j : j + 2])
                    + braille(mat[i : i + 4, j + 2 : j + 4])
                )

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


WEIGHTED_ADJACENCY_TYPES: Final = frozenset({
    "garlaschelli",
    "serrano",
})
"""Names of topologies that generate real-valued (weighted) adjacency matrices."""

ADJACENCY_TYPES: Final = frozenset(
    {
        "barabasialbert",
        "barabasialbert2",
        "barabasialbert4",
        "bus",
        "bus1",
        "bus2",
        "chunglu",
        "configuration",
        "distancedecay",
        "erdosrenyi",
        "feedforward",
        "fractal",
        "garlaschelli",
        # "gapjunctions",
        "lattice",
        "ring",
        "ring1",
        "ring2",
        "serrano",
        "star",
        "startree",
        "strogatzwatts",
    }
    | WEIGHTED_ADJACENCY_TYPES
)
"""Names of the supported adjacency matrix topologies."""


def make_adjacency_matrix_from_name(  # ruff:ignore[too-many-return-statements]
    n: int,
    topology: str,
    *,
    k: int | None = None,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
    """
    :arg k: number of neighbor connections in a ring, bus, or Strogatz-Watts
        network; average number of neighbors in a Erdős-Rényi network; number of
        gap junction clusters; number of levels in a fractal network.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    if dtype is None:
        dtype = xp.float64 if topology in WEIGHTED_ADJACENCY_TYPES else xp.int32

    if rng is None:
        rng = np.random.default_rng()

    if topology == "feedforward":
        return generate_adjacency_feed_forward(n, dtype=dtype, xp=xp)
    elif topology == "ring":
        k = 1 if k is None else k
        return generate_adjacency_ring(n, k=k, dtype=dtype, xp=xp)
    elif topology == "ring1":
        return generate_adjacency_ring(n, k=1, dtype=dtype, xp=xp)
    elif topology == "ring2":
        return generate_adjacency_ring(n, k=2, dtype=dtype, xp=xp)
    elif topology == "bus":
        k = 1 if k is None else k
        return generate_adjacency_bus(n, k=k, dtype=dtype, xp=xp)
    elif topology == "bus1":
        return generate_adjacency_bus(n, k=1, dtype=dtype, xp=xp)
    elif topology == "bus2":
        return generate_adjacency_bus(n, k=2, dtype=dtype, xp=xp)
    elif topology == "star":
        return generate_adjacency_star(n, dtype=dtype, xp=xp)
    elif topology == "startree":
        return generate_adjacency_star_tree(n, dtype=dtype, xp=xp)
    elif topology == "lattice":
        return generate_adjacency_lattice(n, dtype=dtype, xp=xp)
    elif topology == "erdosrenyi":
        return generate_adjacency_erdos_renyi(n, k=k, dtype=dtype, rng=rng, xp=xp)
    elif topology == "strogatzwatts":
        k = 1 if k is None else k
        return generate_adjacency_strogatz_watts(n, k=k, dtype=dtype, rng=rng, xp=xp)
    elif topology == "barabasialbert":
        k = 2 if k is None else k
        return generate_adjacency_barabasi_albert(n, k, dtype=dtype, rng=rng, xp=xp)
    elif topology == "barabasialbert2":
        return generate_adjacency_barabasi_albert(n, 2, dtype=dtype, rng=rng, xp=xp)
    elif topology == "barabasialbert4":
        return generate_adjacency_barabasi_albert(n, 4, dtype=dtype, rng=rng, xp=xp)
    elif topology == "distancedecay":
        return generate_adjacency_distance_decay(n, dtype=dtype, rng=rng, xp=xp)
    elif topology == "configuration":
        return generate_adjacency_configuration(n, dtype=dtype, rng=rng, xp=xp)
    elif topology == "chunglu":
        return generate_adjacency_chung_lu(n, dtype=dtype, rng=rng, xp=xp)
    elif topology == "garlaschelli":
        return generate_weighted_random_graph_garlaschelli(
            n, dtype=dtype, rng=rng, xp=xp
        )
    elif topology == "serrano":
        return generate_weighted_random_graph_serrano(n, dtype=dtype, rng=rng, xp=xp)
    elif topology == "gapjunctions":
        k = max(int(n // 9) - 1, 1) if k is None else k
        return generate_adjacency_gap_junctions(n, k, dtype=dtype, rng=rng, xp=xp)
    elif topology == "fractal":
        # NOTE: for k levels, we have n = p**{k + 1} nodes in the network
        k = 4 if k is None else k
        p = round(n ** (1 / (k + 1)))
        base = "".join(f"{rng.integers(2)}" for _ in range(p))
        return generate_adjacency_fractal(base, nlevels=k, dtype=dtype, xp=xp)
    else:
        raise ValueError(f"unknown topology: '{topology}'")


# }}}


# {{{ graph laplacian


def make_graph_laplacian_undirected(
    A: AArray2D[np.floating[Any]],
    *,
    normalize: bool = False,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
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
    if xp is None:
        xp = array_api_compat.array_namespace(A)

    if A.ndim != 2:
        raise ValueError(f"'A' matrix is not 2D: {A.shape}")

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"'A' matrix is not square: {A.shape}")

    if __debug__ and not xp.allclose(A, A.T):
        raise ValueError("'A' matrix is not symmetric")

    D = xp.sum(A, axis=1)
    L = -A
    xp.fill_diagonal(L, D)
    if normalize:
        Dinv = xp.where(D > 0, 1.0 / xp.sqrt(D), 0.0)
        L = Dinv[:, None] * L * Dinv[None, :]

    return L


def make_graph_laplacian_directed(
    A: AArray2D[np.floating[Any]],
    *,
    out: bool = True,
    normalize: bool = False,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
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
    if xp is None:
        xp = array_api_compat.array_namespace(A)

    L = -A
    if out:
        D = xp.sum(A, axis=1)
        xp.fill_diagonal(L, D)

        if normalize:
            Dinv = xp.where(D > 0, 1.0 / D, 0.0)
            L = Dinv[:, None] * L
    else:
        D = xp.sum(A, axis=0)
        xp.fill_diagonal(L, D)

        if normalize:
            Dinv = xp.where(D > 0, 1.0 / D, 0.0)
            L = L * Dinv[None, :]  # ruff:ignore[non-augmented-assignment]

    assert L.shape == A.shape
    return L


# }}}


# {{{ make_adjacency_from_edges


def make_adjacency_from_edges(
    n: int,
    edges: dict[tuple[int, int], float],
    *,
    symmetrize: bool = False,
    dtype: DTypeLike | None = None,
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
    """Construct an adjacency matrix from a given mapping of edges.

    :arg edges: a mapping from ``(i, j)`` node tuples to the weight of the edge.
    :arg symmetrize: if *True*, both ``(i, j)`` and ``(j, i)`` edges are added.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    result = xp.zeros((n, n), dtype=dtype, device=device)
    for (i, j), w in edges.items():
        if not 0 <= i < n or not 0 <= j < n:
            raise ValueError(f"Edges out of bounds for {n} nodes: ({i}, {j})")

        result[i, j] = w
        if symmetrize:
            result[j, i] = w

    return result


# }}}


# {{{ binary adjacency matrices


def generate_adjacency_all(
    n: int,
    *,
    dtype: DTypeLike | None = None,
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
    r"""Generate a all-to-all :math:`n \times n` adjacency matrix."""
    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    if dtype is None:
        dtype = xp.int32

    result = xp.ones((n, n), dtype=dtype, device=device)
    xp.fill_diagonal(result, 0)

    return result


def generate_adjacency_feed_forward(
    n: int,
    *,
    dtype: DTypeLike | None = None,
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
    r"""Generate a :math:`n \times n` lower triangular adjacency matrix."""
    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    if dtype is None:
        dtype = xp.int32

    result = xp.ones((n, n), dtype=dtype, device=device)
    return xp.tril(result, k=-1)


def generate_adjacency_ring(
    n: int,
    *,
    k: int = 1,
    dtype: DTypeLike | None = None,
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
    """Generate a *k*-ring network with :math:`n` nodes.

    In this network, each node is connected to its :math:`k` nearest neighbors
    with periodicity. For a non-periodic version see :func:`generate_adjacency_bus`.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    if dtype is None:
        dtype = xp.int32

    if n < 0:
        raise ValueError(f"negative dimensions are not allowed: '{n}'")

    if n == 1:
        return xp.zeros((n, n), dtype=dtype)

    if not 0 <= k < n:
        raise ValueError(f"number of neighbors 'm' is invalid: '{k}' (not in [0, {n})")

    # NOTE: this is essentially just a periodic banded matrix
    eye = xp.eye(n, dtype=dtype, device=device)
    result = xp.zeros((n, n), dtype=dtype, device=device)

    for i in range(-k, k + 1):
        if i == 0:
            continue

        result += xp.roll(eye, i, axis=1)

    return result


def generate_adjacency_bus(
    n: int,
    *,
    k: int = 1,
    dtype: DTypeLike | None = None,
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
    """Generate a bus network with :math:`n` nodes.

    In this network, each node is connected to its :math:`k` nearest neighbors
    in a non-periodic fashion.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    if dtype is None:
        dtype = xp.int32

    if n < 0:
        raise ValueError(f"negative dimensions are not allowed: '{n}'")

    if n == 1:
        return xp.zeros((n, n), dtype=dtype)

    if not 0 <= k < n:
        raise ValueError(f"number of neighbors 'k' is invalid: '{k}' (not in [0, {n}])")

    # NOTE: this is essentially just a non-periodic banded matrix
    ones = xp.ones(n, dtype=dtype, device=device)
    result = xp.zeros((n, n), dtype=dtype, device=device)

    for i in range(-k, k + 1):
        if i == 0:
            continue

        result += device.diag(ones[abs(i) :], k=i)

    return result


def generate_adjacency_star(
    n: int,
    *,
    dtype: DTypeLike | None = None,
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
    """Generate a star network with :math:`n` nodes.

    In this network, there is a central node connected to all nodes.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    if dtype is None:
        dtype = xp.int32

    if n < 0:
        raise ValueError(f"negative dimensions are not allowed: '{n}'")

    result = xp.zeros((n, n), dtype=dtype, device=device)
    result[0, 1:] = 1
    result[1:, 0] = 1

    return result


def generate_adjacency_star_tree(
    n: int,
    *,
    nhubs: int | None = None,
    dtype: DTypeLike | None = None,
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
    """Generate a star of stars network with :math:`n` nodes.

    In this setup, the network will have:
    * a central hub node.
    * *nhubs* nodes connected to the central node.
    * the remaining nodes will be equally distributed across the sub-hub nodes.

    :arg nhubs: number of hubs connected to the central hub node. By default,
        this depends on *n*.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    if nhubs is None:
        nhubs = max(n // 5, 1)

    if dtype is None:
        dtype = xp.int32

    if n < 0:
        raise ValueError(f"negative dimensions are not allowed: '{n}'")

    if n == 0 or nhubs == 1:
        return generate_adjacency_star(n, dtype=dtype)

    if nhubs < 0:
        raise ValueError(f"negative number of hubs is not allowed: '{nhubs}'")

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
    partitions = (
        1 + nhubs + xp.linspace(0, n_hub_nodes, nhubs + 1, dtype=dtype, device=device)
    )

    result = xp.zeros((n, n), dtype=dtype, device=device)
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
    dims: int | tuple[int, ...],
    *,
    k: int = 1,
    dtype: DTypeLike | None = None,
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
    r"""Generate an :math:`d`-dimensional lattice adjacency matrix.

    The lattice is defined by *dims*, which can be a single integer (creating
    a 1D bus network) or a tuple of integers defining the size along each
    dimension. For example, ``dims = (n, m)`` creates an :math:`n \times m`
    grid with :math:`n m` nodes.

    Each node is connected to at most :math:`2 k d` neighbors, where
    :math:`d` is the number of dimensions.

    :arg dims: size of the lattice along each dimension. If this is a single
        integer instead of a tuple, a 2D lattice of approximate size
        :math:`\sqrt{n} \times \sqrt{n}` is created.
    :arg k: number of nearest neighbors to connect in each dimension.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    if dtype is None:
        dtype = xp.int32

    if isinstance(dims, int):
        dims = _find_equal_factors(int(dims))

    if not dims:
        raise ValueError("at least one dimension is required")

    if any(d < 0 for d in dims):
        raise ValueError(f"negative dimensions are not allowed: {dims}")

    if k <= 0:
        raise ValueError(f"number of neighbors 'k' must be positive: '{k}'")

    n = 1
    for d in dims:
        n *= d

    result = xp.zeros((n, n), dtype=dtype)
    if n == 1:
        return result

    for i, d in enumerate(dims):
        factors = [xp.eye(dj, dtype=dtype) for dj in dims]
        factors[i] = generate_adjacency_bus(
            d, k=min(k, d - 1), dtype=dtype, device=device, xp=xp
        )

        kron = factors[0]
        for f in factors[1:]:
            kron = xp.kron(kron, f)

        result += kron

    return result


def generate_adjacency_erdos_renyi(
    n: int,
    *,
    p: float | None = None,
    k: int | None = None,
    dtype: DTypeLike | None = None,
    symmetric: bool = True,
    rng: np.random.Generator | None = None,
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
    r"""Generate a random Erdős-Rényi :math:`n \times n` adjacency matrix.

    :arg p: probability of an edge between two nodes (defaults to *0.25*).
    :arg k: average number of edges for each node (i.e. the degree). If *p* is
        not given, it is computed as :math:`p = k / (n - 1)`.
    :arg symmetric: if *True*, the adjacency matrix will be symmetric.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    if n < 0:
        raise ValueError(f"negative dimensions are not allowed: '{n}'")

    if p is not None and k is not None:
        raise ValueError("cannot pass both 'p' and 'k'")

    if p is not None and not 0.0 <= p <= 1.0:
        raise ValueError(f"probability 'p' not in [0, 1]: '{p}'")

    if k is not None and not 0 <= k < n:
        raise ValueError(f"cannot have more edges 'k' than nodes 'n': {k} > {n}")

    if dtype is None:
        dtype = xp.int32

    if n == 1:
        return xp.zeros((n, n), dtype=dtype)

    if p is None:
        p = 0.25 if k is None else (k / (n - 1))

    if p == 0:
        return xp.zeros((n, n), dtype=dtype)

    if rng is None:
        rng = np.random.default_rng()

    if symmetric:
        rows, cols = xp.tril_indices(n, k=-1)
        mask = xp.asarray(rng.random(size=rows.size) < p)

        result = xp.zeros((n, n), dtype=dtype)
        result[rows[mask], cols[mask]] = 1
        result[cols[mask], rows[mask]] = 1
    else:
        # NOTE: setting the diagonal to zero should not change the statistics
        result = xp.astype(xp.asarray(rng.random(size=(n, n)) < p), dtype)
        xp.fill_diagonal(result, 0)

    # NOTE: reconnect any isolated nodes to ensure no zero rows
    for i in range(n):
        if xp.all(result[i] == 0):
            j = rng.choice([j for j in range(n) if j != i])
            result[i, j] = 1
            if symmetric:
                result[j, i] = 1

    return result


def rewire_adjacency_small_world(
    mat: AArray2D[np.floating[Any]],
    *,
    p: float = 0.1,
    forward_only: bool = True,
    rng: np.random.Generator | None = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
    """Rewire edges of an adjacency matrix with probability *p*.

    For each edge, with probability *p*, the edge is removed and replaced by
    a new edge to a randomly chosen non-neighbor. If *forward_only* is
    ``True``, each undirected edge is visited exactly once (from the row
    side); otherwise, all edges of each node are considered. This is known as
    a "small-world" network.

    :arg p: rewiring probability.
    :arg forward_only: if ``True``, only upper-triangle edges are visited.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(mat)

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability 'p' not in [0, 1]: '{p}'")

    if rng is None:
        rng = np.random.default_rng()

    assert mat.ndim == 2
    assert mat.shape[0] == mat.shape[1]

    result = xp.copy(mat)
    n = result.shape[0]

    for i in range(n):
        if forward_only:
            edges = xp.flatnonzero(result[i, i + 1 :]) + i + 1
        else:
            edges = xp.flatnonzero(result[i])

        for j in edges:
            if not rng.random() < p:
                continue

            # find all candidate edges to rewrite
            connected = result[i] != 0
            connected[i] = True
            choices = xp.flatnonzero(~connected)

            if choices.size == 0:
                continue

            # remove the old edge
            result[i, j] = result[j, i] = 0

            # rewire the edge randomly
            jnew = rng.choice(choices)
            result[i, jnew] = result[jnew, i] = 1

    return result


def generate_adjacency_strogatz_watts(
    n: int,
    *,
    k: int = 2,
    p: float = 0.1,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
    r"""Generate a random Strogatz-Watts :math:`n \times n` adjacency matrix.

    :arg k: number of neighboring nodes.
    :arg p: rewiring probability.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability 'p' not in [0, 1]: '{p}'")

    if dtype is None:
        dtype = xp.int32

    if rng is None:
        rng = np.random.default_rng()

    result = generate_adjacency_ring(n, k=k, dtype=dtype, device=device, xp=xp)
    return rewire_adjacency_small_world(result, p=p, forward_only=True, rng=rng, xp=xp)


def generate_adjacency_barabasi_albert(
    n: int,
    m: int,
    *,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
    """Generate a random Barabási-Albert adjacency matrix.

    :arg m: number of edges each new node should attach to :math:`m < n`.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    if n < 0:
        raise ValueError(f"negative dimensions are not allowed: '{n}'")

    if m < 0:
        raise ValueError(f"negative number of edges is not allowed: '{m}'")

    if m >= n:
        raise ValueError(f"invalid sizes (m >= n): {m} >= {n}")

    if dtype is None:
        dtype = xp.int32

    if rng is None:
        rng = np.random.default_rng()

    mat = xp.zeros((n, n), dtype=dtype, device=device)

    # construct an initial dense block of m + 1 nodes with m edges
    mat[: m + 1, : m + 1] = 1
    xp.fill_diagonal(mat, 0)

    degrees = xp.zeros(n, dtype=xp.int64, device=device)
    degrees[: m + 1] = m

    # iteratively add more nodes
    for i in range(m + 1, n):
        # construct probability based on degree: larger degree nodes get a larger
        # probability of additional connections => even bigger group!
        d = xp.sum(mat[:i, :i], axis=1)
        p = d / xp.sum(d)

        # randomly choose some connections
        j = rng.choice(i, size=m, replace=False, p=p)
        mat[i, j] = 1
        mat[j, i] = 1

        degrees[i] += m
        degrees[j] += 1

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
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
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
        farther away vertices is. Defaults to the domain diagonal divided by
        ``log(max(n, 4))``, which scales with the number of vertices to avoid
        isolated nodes for small graphs.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    if n < 0:
        raise ValueError(f"negative dimensions are not allowed: '{n}'")

    if not 0.0 < beta <= 1.0:
        raise ValueError(f"'beta' must be in (0, 1]: {beta}")

    if xlim[0] > xlim[1]:
        xlim = (xlim[1], xlim[0])

    if ylim is None:
        ylim = xlim

    if ylim[0] > ylim[1]:
        ylim = (ylim[1], ylim[0])

    if sigma is None:
        sigma = 1.4 * xp.hypot(xlim[1] - xlim[0], ylim[1] - ylim[0]) / xp.log(max(n, 4))

    if sigma <= 0.0:
        raise ValueError(f"'sigma' cannot be negative: {sigma}")

    if dtype is None:
        dtype = xp.int32

    if rng is None:
        rng = np.random.default_rng()

    # generate some random points
    x = xp.stack([
        xp.asarray(rng.uniform(xlim[0], xlim[1], size=n), device=device),
        xp.asarray(rng.uniform(ylim[0], ylim[1], size=n), device=device),
    ])
    d = xp.sqrt(xp.sum((x[:, None, :] - x[:, :, None]) ** 2, axis=0))

    # probabilities
    p = beta * xp.exp(-d / sigma)

    # adjacency
    mat = rng.random((n, n))
    mat = (mat < p).astype(dtype)

    xp.fill_diagonal(mat, 0)
    if symmetric:
        tril = xp.tril_indices(n, k=-1)
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
    device: Any = None,
    xp: Any = None,
) -> AArray1D[np.integer[Any]]:
    # FIXME: this seems like it'll have mean *mean* only if n > mean * m?
    smax = min(n - 1, avgsize * m)
    for _ in range(maxiter):
        # generate candidates
        p = xp.asarray(rng.dirichlet((alpha,) * m))
        x = xp.astype(xp.rint(p * smax), xp.int32)

        # ensure they sum up to smax
        extra = smax - xp.sum(x)
        if extra != 0:
            idx = rng.choice(m, size=abs(extra), replace=True)
            x[idx] += xp.sign(extra)

        # check that the maximum size is respected
        if xp.max(x) <= maxsize and xp.min(x) >= 1:
            break

    return x


def _make_adjacency_from_groups(
    groups: AArray1D[np.integer[Any]],
    gaps: int | AArray1D[np.integer[Any]],
    *,
    dtype: DTypeLike | None = None,
    xp: Any = None,
) -> tuple[
    AArray2D[np.floating[Any]], AArray1D[np.integer[Any]], AArray1D[np.integer[Any]]
]:
    device = array_api_compat.device(groups)
    if dtype is None:
        dtype = xp.int32

    if isinstance(gaps, int):
        if xp is None:
            xp = array_api_compat.array_namespace(groups)

        gaps = xp.asarray([gaps] * groups.size, device=device)

    if xp is None:
        xp = array_api_compat.array_namespace(groups, gaps)

    n = int(xp.sum(groups) + xp.sum(gaps))
    if groups.shape != gaps.shape:
        raise ValueError(
            "cluster sizes and gap sizes must have the same shape: "
            f"got {groups.shape} and {gaps.shape}"
        )

    i = 0
    result = xp.zeros((n, n), dtype=dtype, device=device)

    for m, g in zip(groups, gaps, strict=True):
        result[i : i + m, i : i + m] = 1.0
        i += m + g

    xp.fill_diagonal(result, 0)
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
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
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
    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    if n < 0:
        raise ValueError(f"negative dimensions are not allowed: '{n}'")

    if m < 0:
        raise ValueError(f"negative cluster counts are not allowed: '{m}'")

    if dtype is None:
        dtype = xp.int32

    if rng is None:
        rng = np.random.default_rng()

    if avgsize > maxsize:
        raise ValueError(
            f"'avgsize' cannot be larger than 'maxsize': {avgsize} > {maxsize}"
        )

    if m == 0 or n <= 1:
        return xp.zeros((n, n), dtype=dtype, device=device)

    # generate gap junction clusters
    groups = _generate_random_gap_junction_clusters(
        rng,
        n,
        m,
        alpha=alpha,
        avgsize=avgsize,
        maxsize=maxsize,
        maxiter=maxiter,
        device=device,
        xp=xp,
    )
    leftover = n - xp.sum(groups)

    # generate random gaps
    cuts = xp.asarray(
        rng.choice(np.arange(1, leftover), size=m - 1, replace=False), device=device
    )
    pts = xp.concatenate(([0], xp.sort(cuts), [leftover]))
    gaps = xp.diff(pts)
    assert (xp.sum(groups) + xp.sum(gaps)) == n

    # create adjacency matrix
    result, _, _ = _make_adjacency_from_groups(groups, gaps, dtype=dtype)
    assert result.shape == (n, n)

    return result


def _expand_pattern(
    base: str, nlevels: int, dtype: DTypeLike | None = None, *, xp: Any
) -> AArray1D[np.floating[Any]]:
    zeros = "0" * len(base)
    pattern = base
    for _ in range(nlevels - 1):
        pattern = "".join([(base if i == "1" else zeros) for i in pattern])

    # transform pattern to 0/1 array
    return xp.asarray(np.fromiter(pattern, dtype=dtype))


def generate_adjacency_fractal(
    base: str,
    *,
    nlevels: int = 4,
    dtype: DTypeLike | None = None,
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
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
    if xp is None:
        xp = array_api_compat.array_namespace(np.empty(0))

    if not set(base) <= {"0", "1"}:
        raise ValueError(f"'base' pattern must be binary only: '{base}'")

    if nlevels < 0:
        raise ValueError(f"'nlevels' must be non-negative: {nlevels}")

    if dtype is None:
        dtype = xp.int32

    if nlevels == 0:
        # FIXME: what size do we actually expect here? this chose was mostly
        # made to fit match `len(base) ** 0`
        return xp.zeros((1, 1), dtype=dtype, device=device)

    from scipy.linalg import circulant

    x = _expand_pattern(base, nlevels, dtype=dtype, xp=xp)
    mat = circulant(x).T
    xp.fill_diagonal(mat, 0)

    return mat


def generate_adjacency_astrocyte_lattice(
    points: AArray2D[np.floating[Any]],
    *,
    variant: Literal[
        "regular-degree", "link-radius", "shortcut", "scale-free", "erdos-renyi"
    ] = "regular-degree",
    k_nearest_neighbors: int = 4,
    max_neighbor_distance: float = 1.0,
    p: float = 0.2,
    m_sf: int = 2,
    rc: float = 1.0,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
    """Generate lattice-based astrocyte networks from [Lallouette2014]_.

    The supported variants are described in [Lallouette2014]_. Briefly, they are

    * ``regular-degree``: constructs a graph with edges between k nearest neighbors
      based on *points*.
    * ``link-radius``: constructs a graph with edges between nodes within
      *max_neighbor_distance* based on *points*.
    * ``shortcut``: constructs a Strogatz-Watts small-world type network on a lattice.
    * ``scale-free``: constructs Barabási-Albert type network on a lattice.
    * ``erdos-renyi``: equivalent to :func:`generate_adjacency_erdos_renyi`.

    :arg k_nearest_neighbors: the number of nearest neighbors to consider when
        constructing the "regular-degree" network.
    :arg max_neighbor_distance: the maximum distance between neighbors when
        constructing the "link-radius" network.
    :arg p: the coupling probability used when constructing the "shortcut" or the
        "erdos-renyi" networks.
    :arg m_sf: number of links added in the "scale-free" network.
    :arg rc: spatial decay constant used in the "scale-free" network.
    """
    if not array_api_compat.is_numpy(points):
        raise TypeError(
            "'generate_adjacency_astrocyte_lattice' only supports 'numpy' arrays: "
            f"{type(points)}"
        )

    if xp is None:
        xp = array_api_compat.array_namespace(points)

    device = array_api_compat.device(points)
    if rng is None:
        rng = np.random.default_rng()

    if points.ndim != 2:
        raise ValueError(f"'points' must be a 2D array: {points.shape}")

    n, dim = points.shape
    if dim < 1:
        raise ValueError(f"'points' must have at least one dimension: {points.shape}")

    if n <= 1:
        return xp.zeros((n, n), dtype=dtype, device=device)

    m_latt = int(n ** (1.0 / dim))
    if not 0 < k_nearest_neighbors <= n:
        raise ValueError(
            f"'k_nearest_neighbors' must be positive < n: {k_nearest_neighbors}"
        )

    if max_neighbor_distance <= 0:
        raise ValueError(
            f"'max_neighbor_distance' must be positive: {max_neighbor_distance}"
        )

    if not 0 <= p <= 1.0:
        raise ValueError(f"'p' probability must be in [0, 1]: {p}")

    if not 0 < m_sf < n:
        raise ValueError(f"'m_sf' link count must be positive: {m_sf}")

    if rc <= 0.0:
        raise ValueError(f"'rc' must be positive: {rc}")

    if dtype is None:
        dtype = xp.int32

    if variant == "regular-degree":
        from scipy.spatial import KDTree

        tree = KDTree(points)
        _, indices = tree.query(points, k=k_nearest_neighbors + 1)
        indices = indices[:, 1:]

        mat = xp.zeros((n, n), dtype=dtype, device=device)
        for i, j in enumerate(indices):
            mat[i, j] = 1
            mat[j, i] = 1
    elif variant == "link-radius":
        from scipy.spatial import KDTree

        tree = KDTree(points)
        indices = tree.query_ball_point(points, r=max_neighbor_distance)

        mat = xp.zeros((n, n), dtype=dtype, device=device)
        for i, nbrs in enumerate(indices):
            # filter out self
            j = xp.array(nbrs, device=device)
            j = j[j != i]

            mat[i, j] = 1
            mat[j, i] = 1
    elif variant == "shortcut":
        # NOTE: this is basically a Strogatz-Watts on a lattice
        if m_latt**dim != n:
            raise ValueError(
                f"variant='shortcut' expects grid-like 'points': {dim} does not "
                f"match {n} exactly"
            )

        mat = generate_adjacency_lattice(
            (m_latt,) * dim,
            k=k_nearest_neighbors,
            dtype=dtype,
        )
        mat = rewire_adjacency_small_world(mat, p=p, rng=rng)
    elif variant == "scale-free":
        # NOTE: this is basically a Barabási-Albert on a lattice with slightly
        # fancier probabilities: they depend on degree and distance
        mat = xp.zeros((n, n), dtype=dtype, device=device)

        # fully connect the first m nodes
        mat[: m_sf + 1, : m_sf + 1] = 1
        xp.fill_diagonal(mat[: m_sf + 1, : m_sf + 1], 0)

        # keep track of degrees
        degrees = xp.zeros(n, dtype=xp.int64, device=device)
        degrees[: m_sf + 1] = m_sf

        eps = 100 * xp.finfo(points.dtype).eps
        for i in range(m_sf + 1, n):
            # construct probability based on degree:
            #   p_i ~ degree * exp(-||x_j - x_i|| / rc)
            dists = xp.linalg.norm(points[:i] - points[i], axis=1)
            weights = degrees[:i] * xp.exp(-dists / rc)

            total = xp.sum(weights)
            if total < eps:  # ruff:ignore[if-else-block-instead-of-if-exp]
                probability = xp.ones(i) / i
            else:
                probability = weights / total

            # choose some connections based on these probabilities
            j = rng.choice(i, size=m_sf, replace=False, p=probability)
            mat[i, j] = 1
            mat[j, i] = 1

            degrees[i] += m_sf
            degrees[j] += 1
    elif variant == "erdos-renyi":
        mat = generate_adjacency_erdos_renyi(n, p=p, dtype=dtype, rng=rng)
    else:
        raise ValueError(f"unsupported adjacency variant: {variant}")

    return mat


def generate_adjacency_configuration(
    n: int,
    *,
    degrees: AArray1D[np.integer[Any]] | int | None = None,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
    device: Any = None,
    xp: Any = None,
) -> AArray2D[np.floating[Any]]:
    """Generate a random :math:`n \times n` adjacency matrix for a configuration
    model.

    :arg degrees: an array of desired node degrees that should sum up to an even
        number.
    """
    if n < 0:
        raise ValueError(f"negative dimensions are not allowed: '{n}'")

    if dtype is None:
        dtype = np.int32
    dtype = np.dtype(dtype)

    if n <= 1:
        return np.zeros((n, n), dtype=dtype)

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

    # NOTE: reconnect any isolated nodes to ensure no zero rows
    for i in range(n):
        if np.all(result[i] == 0):
            j = rng.choice([j for j in range(n) if j != i])
            result[i, j] = result[j, i] = 1

    return result


def generate_adjacency_stochastic_block_model(
    n: int,
    *,
    p: float | tuple[float, float] | Array2D[np.floating[Any]] = 0.75,
    labels: int | Array1D[np.integer[Any]] | None = None,
    signed: bool = False,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    r"""Generate an adjacency matrix for a Stochastic Block Model.

    The graph blocks defined by the *labels* node to block assignment. Edges
    within the blocks appear with probability *p*. Note that setting :math:`p =
    (p, p)` (same probability for in-block and between-block) and *signed* to *True*
    will just recover the Erdős-Rényi model.

    :arg labels: a label for each node: (1) If None, then we take *m*
        approximately :math:`\sqrt{n}`; (2) if integer *m*, then it should exactly
        divide *n* and the labels are ``arange(n) // (n // m)``; (3) if an array,
        then each ``labels[i]`` should define the block it belongs to.
    :arg p: edge probabilities: (1) if float, :math:`p` in the same block and
        :math:`(1 - p)` between blocks; (2) if a tuple, then :math:`p[0]` in the
        same block and :math:`p[1]` between blocks; (3) if a symmetric matrix,
        then it should have a size :math:`m \times m` and each entry is the
        probability between a block *i* and *j*.
    :arg signed: if *True*, the the edges between blocks get a value of -1.
    """
    if n < 0:
        raise ValueError(f"negative dimensions are not allowed: '{n}'")

    if dtype is None:
        dtype = np.int32
    dtype = np.dtype(dtype)

    if n <= 1:
        return np.zeros((n, n), dtype=dtype)

    if rng is None:
        rng = np.random.default_rng()

    # {{{ labels

    if labels is None:
        # NOTE: by default we try to do about "m" blocks with "m" elements per block
        m = min(*_find_equal_factors(n))
        labels = np.arange(n) // (n // m)
    elif isinstance(labels, int):
        m = labels
        if m <= 0:
            raise ValueError(f"block labels 'labels' must be positive: '{m}'")

        if n % m != 0:
            raise ValueError(f"'n' must be divisible by 'labels': n={n} and m={m}")

        labels = np.arange(n) // (n // m)
    elif isinstance(labels, np.ndarray):
        if labels.shape != (n,):
            raise ValueError(
                f"'labels' has incorrect shape: {labels.shape} (expected ({n},))"
            )

        # FIXME: does this need to be contiguous? probably not..
        m = int(np.max(labels)) + 1
        if not np.array_equal(np.unique(labels), np.arange(m)):
            raise ValueError("'labels' must be contiguous integers starting from 0")
    else:
        raise TypeError(f"'labels' must be int, ndarray, or None: {type(labels)}")

    # }}}

    # {{{ probabilities

    if isinstance(p, (int, float)):
        q = p
        p = np.full((m, m), 1 - q, dtype=np.float64)
        np.fill_diagonal(p, q)
    elif isinstance(p, tuple):
        if len(p) != 2:
            raise ValueError(f"unsupported tuple for 'p': {p}")

        q = p[0]
        p = np.full((m, m), p[1], dtype=np.float64)
        np.fill_diagonal(p, q)
    elif isinstance(p, np.ndarray):
        if p.shape != (m, m):
            raise ValueError(f"'p' matrix must have shape ({m}, {m}): {p.shape}")

        if not np.allclose(p, p.T):
            raise ValueError("'p' matrix must be symmetric")
    else:
        raise TypeError(f"'p' must be a float, tuple or ndarray: {type(p)}")

    if np.min(p) < 0.0 or np.max(p) > 1.0:
        raise ValueError("probability 'p' not in [0, 1]")

    # }}}

    # {{{ generate

    # create lower triangular blocks
    rows, cols = np.tril_indices(n, k=-1)
    block_i = labels[rows]
    block_j = labels[cols]
    is_same = block_i == block_j

    # create mask based on probabilities
    r = rng.random(size=rows.size)
    mask = r < p[block_i, block_j]

    # set up adjacency
    result = np.zeros((n, n), dtype=dtype)
    if signed:
        same_mask = is_same & mask
        diff_mask = ~is_same & mask
        result[rows[same_mask], cols[same_mask]] = 1
        result[cols[same_mask], rows[same_mask]] = 1
        result[rows[diff_mask], cols[diff_mask]] = -1
        result[cols[diff_mask], rows[diff_mask]] = -1
    else:
        result[rows[mask], cols[mask]] = 1
        result[cols[mask], rows[mask]] = 1

    # }}}

    # make sure the groups are connected
    for i in range(n):
        if np.all(result[i] == 0):
            # NOTE: isolated nodes are always added within the block
            j = rng.choice([k for k in range(n) if k != i and labels[i] == labels[k]])
            result[i, j] = 1
            result[j, i] = 1

    return result


def _make_chung_lu_power_law_degrees(
    n: int,
    *,
    gamma: float = 2.5,
    mu: float = 5.0,
) -> Array1D[np.floating[Any]]:
    p = 1.0 / (gamma - 1.0)
    idx = np.arange(1, n + 1) ** (-p)
    return n * mu * idx / np.sum(idx)


def _sample_chung_lu_edges(
    degrees: Array1D[np.floating[Any]],
    *,
    rng: np.random.Generator,
) -> tuple[Array1D[np.integer[Any]], Array1D[np.integer[Any]]]:
    n = len(degrees)
    total = np.sum(degrees)
    if total < 10 * np.finfo(degrees.dtype).eps:
        empty = np.array([], dtype=np.int64)
        return empty, empty

    iu, ju = np.triu_indices(n, k=1)
    deg_scaled = degrees * (1.0 / total)
    p = deg_scaled[iu] * degrees[ju]
    connected = rng.random(len(iu)) < p
    return iu[connected], ju[connected]


def generate_adjacency_chung_lu(
    n: int,
    *,
    degrees: Array1D[np.floating[Any]] | None = None,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    """Generate a random Chung-Lu adjacency matrix as described in [Chung2002]_.

    The generated graph will have the given *degrees* on average. If the degrees
    are all equal, then this is equivalent to the Erdős-Rényi construction.

    .. [Chung2002] F. Chung, L. Lu,
        *Connected Components in Random Graphs With Given Expected Degree Sequences*,
        Annals of Combinatorics, Vol. 6, pp. 125--145, 2002,
        `doi:10.1007/pl00012580 <https://doi.org/10.1007/pl00012580>`__.

    :arg degrees: an array of expected node degrees, which are not necessarily
        integer values. If no degrees are given, then a simple power law with
        mean degree *5.0* is used.
    """

    if n < 0:
        raise ValueError(f"negative dimensions are not allowed: '{n}'")

    if dtype is None:
        dtype = np.int32
    dtype = np.dtype(dtype)

    if n <= 1:
        return np.zeros((n, n), dtype=dtype)

    if rng is None:
        rng = np.random.default_rng()

    if degrees is None:
        # NOTE: we make a little default degree sequence to match [Chung2002]
        degrees = _make_chung_lu_power_law_degrees(n)
    else:
        if degrees.shape != (n,):
            raise ValueError(f"'degrees' must have shape ({n},): {degrees.shape}")

        if np.any(degrees < 0):
            raise ValueError("'degrees' must be non-negative")

    ic, jc = _sample_chung_lu_edges(degrees, rng=rng)

    A = np.zeros((n, n), dtype=dtype)
    A[ic, jc] = 1
    A[jc, ic] = 1

    return A


# }}}


# {{{ weighted adjacency matrices


def generate_weighted_random_graph_garlaschelli(
    n: int,
    *,
    p: float | None = None,
    omega: float | None = None,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    r"""Generate a Weighted Random Graph (WRG) as described in [Garlaschelli2009]_.

    The WRG from [Garlaschelli2009]_ is an extension of the Erdős-Rényi
    construction to weighted graphs. If the adjacency matrix from this is
    extracted, it will be equivalent to the Erdős-Rényi adjacency
    (statistically, of course).

    By construction, the resulting WRG will have integer weights in :math:`\{
    0, 1, \dots\}` drawn from a geometric distribution. The expected edge
    weight is given by

    .. math::

        \omega = \frac{p}{1 - p} \implies p = \frac{\omega}{1 + \omega}.

    .. [Garlaschelli2009] D. Garlaschelli,
        *The Weighted Random Graph Model*,
        New Journal of Physics, Vol. 11, pp. 73005--73005, 2009,
        `doi:10.1088/1367-2630/11/7/073005 <https://doi.org/10.1088/1367-2630/11/7/073005>`__.

    :arg p: geometric parameter in [0, 1) (default: 0.25).
    :arg omega: target mean weight, which must be positive (so that :math:`p \ge 0`)
        and finite (so that :math:`p < 1`).
    """
    import math

    if n < 0:
        raise ValueError(f"negative dimensions are not allowed: '{n}'")

    if p is not None and omega is not None:
        raise ValueError("cannot pass both 'p' and 'omega'")

    if p is not None and not 0.0 <= p < 1.0:
        raise ValueError(f"probability 'p' not in [0, 1): '{p}'")

    if omega is not None and (omega < 0 or not math.isfinite(omega)):
        raise ValueError(
            f"mean edge weight 'omega' must be positive and finite: '{omega}'"
        )

    if p is None:
        p = 0.25 if omega is None else (omega / (1 + omega))

    if dtype is None:
        dtype = np.int32
    dtype = np.dtype(dtype)

    if n <= 1:
        return np.zeros((n, n), dtype=dtype)

    if p == 0:
        return np.zeros((n, n), dtype=dtype)

    if rng is None:
        rng = np.random.default_rng()

    weights = rng.geometric(1 - p, size=n * (n - 1) // 2) - 1

    idx = np.triu_indices(n, k=1)
    result = np.zeros((n, n), dtype=dtype)
    result[idx] = weights
    result += result.T

    # FIXME: ensure that this is actually connected
    return result


def _make_serrano_scaling(
    degrees: Array1D[np.floating[Any]],
    strengths: Array1D[np.floating[Any]],
) -> tuple[float, Array1D[np.floating[Any]]]:
    u = np.zeros_like(degrees, dtype=np.float64)
    nonzero = degrees > 0
    u[nonzero] = strengths[nonzero] / degrees[nonzero]

    mean_k = np.mean(degrees)
    mean_s = np.mean(strengths)
    scale = float(mean_k / mean_s) if mean_s > 0 else 1.0
    return scale, u


def generate_weighted_random_graph_serrano(
    n: int,
    *,
    degrees: Array1D[np.floating[Any]] | None = None,
    strengths: Array1D[np.floating[Any]] | None = None,
    beta: float = 1.5,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    r"""Generate a Weighted Complex Network (WCN) as described in [Serrano2006]_.

    This construction acts as a maximally random null model for weighted
    complex networks with prescribed degree-strength distributions, filtering out
    spurious non-structural correlations. The unweighted graph is generated
    via the Chung-Lu ensemble, and edges are assigned weights according to

    .. math::

        w_{ij} = \frac{\langle k \rangle}{\langle s \rangle} \frac{s_i s_j}{k_i k_j}.

    .. [Serrano2006] M. Á. Serrano, M. Boguñá, R. Pastor-Satorras,
        *Correlations in Weighted Networks*,
        Physical Review E, Vol. 74, pp. 55101--55101, 2006,
        `doi:10.1103/physreve.74.055101 <https://doi.org/10.1103/physreve.74.055101>`__.

    :arg degrees: an array of expected node degrees, which are not necessarily
        integer values. If no degrees are given, then a simple power law with
        mean degree *5.0* is used.
    :arg strengths: an array of expected node strengths. If not given, then
        they are defined as :math:`s_i = k_i^\beta` from the degree distribution.
    """

    if n < 0:
        raise ValueError(f"negative dimensions are not allowed: '{n}'")

    if dtype is None:
        dtype = np.float64
    dtype = np.dtype(dtype)

    if n <= 1:
        return np.zeros((n, n), dtype=dtype)

    if degrees is None:
        degrees = _make_chung_lu_power_law_degrees(n)
    else:
        if degrees.shape != (n,):
            raise ValueError(f"'degrees' must have shape ({n},): {degrees.shape}")

        if np.any(degrees < 0):
            raise ValueError("'degrees' must be non-negative")

    if strengths is None:
        if beta <= 0.0:
            raise ValueError(f"'beta' must be positive: {beta}")

        strengths = degrees**beta
    else:
        if strengths.shape != (n,):
            raise ValueError(f"'strengths' must have shape ({n},): {strengths.shape}")

        if np.any(strengths < 0):
            raise ValueError("'strengths' must be non-negative")

    if rng is None:
        rng = np.random.default_rng()

    ic, jc = _sample_chung_lu_edges(degrees, rng=rng)
    scale, u = _make_serrano_scaling(degrees, strengths)

    w = scale * u[ic] * u[jc]
    W = np.zeros((n, n), dtype=dtype)
    W[ic, jc] = w
    W[jc, ic] = w

    return W


# }}}


# {{{ adding weights


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
    x = rng.multivariate_normal(
        np.zeros(2, dtype=Sigma.dtype),
        Sigma,
        size=mat.shape[0],
    )

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
    r"""Generate random weights for the adjacency matrix *mat* with equal row sum.

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
            raise ValueError(f"row {i} has no connections")

        # NOTE: the weights are guaranteed to not be zero here due to the
        # Dirichlet distribution, but they can be arbitrarily small nonetheless.
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
    f: Callable[[Array1D[np.floating[Any]]], Array1D[np.floating[Any]]],
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
    f: Callable[[Array1D[np.floating[Any]]], Array1D[np.floating[Any]]],
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


def generate_random_degree_proportional_weights(
    mat: Array2D[np.floating[Any]],
    *,
    alpha: float = 1.0,
    dtype: DTypeLike | None = None,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    r"""Generate random weights for the adjacency matrix *mat* proportional
    to each node's degree.

    For this, we generate a random matrix with equal row sum (see
    :func:`generate_random_equal_row_sum`) and then scale each row based on its
    degree. Therefore, nodes with a higher degree will "receive" proportional
    inputs from their neighbors. The scaling is taken as

    .. math::

        W_{ij} = \frac{d_i^\alpha}{\max d_i^\alpha} W_{ij}

    where :math:`d_i` is the degree of node :math:`i`. Using this scaling, we can
    obtain

    * :math:`\alpha = 0`: the standard equal row sum weights.
    * :math:`\alpha = 1`: row sums proportional to degree.
    * :math:`0 < \alpha < 1`: a sublinear scaling with respect to degree.
    * :math:`\alpha < 0`: high-degree nodes receive less input.
    * :math:`\alpha > 1`: high-degree nodes are amplified.
    """

    result = generate_random_equal_row_sum(mat, dtype=dtype, rng=rng)
    degree = np.sum(mat, axis=1).astype(dtype)

    # handle isolated nodes
    isolated = degree == 0
    degree[isolated] = 1

    scaling = degree**alpha
    scaling[isolated] = 0
    max_scaling = np.max(scaling)

    if max_scaling > 0:
        scaling /= max_scaling

    result *= scaling.reshape(-1, 1)
    return result


def generate_random_serrano_weights(
    mat: Array2D[np.floating[Any]],
    *,
    degrees: Array1D[np.floating[Any]] | None = None,
    strengths: Array1D[np.floating[Any]] | None = None,
    beta: float = 1.5,
    dtype: DTypeLike | None = None,
) -> Array2D[np.floating[Any]]:
    r"""Generate edge weights for *mat* according to [Serrano2006]_.

    This assigns weights to the existing edges of the adjacency matrix *mat*
    such that edge weights follow the uncorrelated baseline:

    .. math::

        w_{ij} = \frac{\langle k \rangle}{\langle s \rangle} \frac{s_i s_j}{k_i k_j}.

    :arg degrees: an array of node degrees. If not given, degrees are computed
        from *mat* as the row counts of non-zero entries.
    :arg strengths: an array of expected node strengths. If not given, then
        they are defined as :math:`s_i = k_i^\beta` from the degree distribution.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"adjacency matrix should be square: {mat.shape}")

    if dtype is None:
        dtype = np.float64
    dtype = np.dtype(dtype)

    n = mat.shape[0]
    if n <= 1:
        return np.zeros(mat.shape, dtype=dtype)

    if degrees is None:
        degrees = np.count_nonzero(mat != 0, axis=1).astype(np.float64)
    else:
        if degrees.shape != (n,):
            raise ValueError(f"'degrees' must have shape ({n},): {degrees.shape}")

        if np.any(degrees < 0):
            raise ValueError("'degrees' must be non-negative")

    if strengths is None:
        if beta <= 0.0:
            raise ValueError(f"'beta' must be positive: {beta}")

        strengths = degrees**beta
    else:
        if strengths.shape != (n,):
            raise ValueError(f"'strengths' must have shape ({n},): {strengths.shape}")
        if np.any(strengths < 0):
            raise ValueError("'strengths' must be non-negative")

    iu, ju = np.nonzero(np.triu(mat != 0, k=1))
    scale, u = _make_serrano_scaling(degrees, strengths)

    w = scale * u[iu] * u[ju]
    result = np.zeros(mat.shape, dtype=dtype)
    result[iu, ju] = w
    result[ju, iu] = w

    return result


# }}}


# {{{ shuffle_adjacency_weights


def shuffle_adjacency_local_weights(
    mat: Array2D[np.floating[Any]],
    *,
    eps: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    """Shuffle weights over node-wise edges of an adjacency matrix *mat*.

    This function shuffles non-zero edge weights for each node. As such, it will
    preserve the strength / degree of the node, but it will result in an
    asymmetric matrix (i.e. a directed graph).
    """
    if eps < 0.0:
        raise ValueError(f"'eps' must be non-negative: {eps}")

    if rng is None:
        rng = np.random.default_rng()

    mask = np.abs(mat) > eps

    result = np.copy(mat)
    for i in range(mat.shape[0]):
        nz = np.flatnonzero(mask[i])
        if len(nz) > 1:
            result[i, nz] = rng.permutation(result[i, nz])

    return result


def shuffle_adjacency_global_weights(
    mat: Array2D[np.floating[Any]],
    *,
    eps: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    """Shuffle weights over all edges of a symmetric *mat*.

    This function shuffles non-zero weights globally. This will preserve the
    topology of the graph represented by *mat*, but not most other measures.
    The resulting graph will undirected.
    """
    if eps < 0.0:
        raise ValueError(f"'eps' must be non-negative: {eps}")

    if rng is None:
        rng = np.random.default_rng()

    # get upper triangular values from the adjacency
    iu, ju = np.nonzero(np.triu(np.abs(mat) > eps, k=1))
    weights = rng.permutation(mat[iu, ju])

    # chug them in a new matrix
    result = np.copy(mat)
    result[iu, ju] = weights
    result[ju, iu] = weights

    return result


def rewire_adjacency(
    mat: Array2D[np.floating[Any]],
    *,
    q: int = 10,
    eps: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Array2D[np.floating[Any]]:
    """Rewire the edges in *mat* such that the node degree is preserved.

    This performs a standard degree-preserving double-edge swap (Maslov-Sneppen)
    on an undirected graph.
    """
    if eps < 0.0:
        raise ValueError(f"'eps' must be non-negative: {eps}")

    if q < 1:
        raise ValueError(f"'q' rewiring factor must be >= 1: {q}")

    if rng is None:
        rng = np.random.default_rng()

    result = np.copy(mat)

    # get edges
    u_, v_ = np.nonzero(np.triu(np.abs(mat) > eps, k=1))
    m = len(u_)
    if m <= 2:
        return result

    # NOTE: turns out creating np.int64s every time we access this is expensive
    u = memoryview(u_)
    v = memoryview(v_)

    n_attempts = q * m
    batch_size = min(n_attempts, 20_000)
    attempts_remaining = n_attempts

    while attempts_remaining > 0:
        chunk = min(attempts_remaining, batch_size)
        idx1 = memoryview(rng.integers(m, size=chunk))
        idx2 = memoryview(rng.integers(m, size=chunk))
        flip = memoryview(rng.random(chunk) < 0.5)
        attempts_remaining -= chunk

        for k in range(chunk):
            i = idx1[k]
            j = idx2[k]
            if i == j:
                continue

            u1 = u[i]
            v1 = v[i]
            if flip[k]:
                u2 = u[j]
                v2 = v[j]
            else:
                u2 = u[j]
                v2 = v[j]

            # skip repeated edges or ones that already exist
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:  # ruff: ignore[compare-with-tuple, repeated-equality-comparison]
                continue

            if abs(result[u1, v2]) > eps or abs(result[u2, v1]) > eps:
                continue

            # swap edges in result
            w1 = result[u1, v1]
            w2 = result[u2, v2]
            result[u1, v1] = result[v1, u1] = 0.0
            result[u2, v2] = result[v2, u2] = 0.0
            result[u1, v2] = result[v2, u1] = w1
            result[u2, v1] = result[v1, u2] = w2

            # update edges
            u[i] = u1
            v[i] = v2
            u[j] = u2
            v[j] = v1

    return result


# }}}
