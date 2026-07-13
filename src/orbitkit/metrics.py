# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from orbitkit.typing import Array1D, Array2D
from orbitkit.utils import module_logger

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

log = module_logger(__name__)

# {{{ compute_weighted_degree


def compute_weighted_degree(
    mat: Array2D[np.floating[Any]],
) -> Array1D[np.floating[Any]]:
    r"""Compute the weighted degree (or strength) of each node in the graph.

    .. math::

        s_i = \sum_{j}^n W_{ij}.

    Note that, by definition, this also works for signed networks. However,
    cancellation can occur if the weights balance out, so a strength of 0 does
    not mean that the node is isolated.
    """

    n, m = mat.shape
    if n != m:
        raise ValueError(f"matrix not square: {mat.shape}")

    return np.sum(mat, axis=1)


# }}}


# {{{ compute_weighted_clustering_coefficient_barrat


def compute_weighted_clustering_coefficient_barrat(
    mat: Array2D[np.floating[Any]],
    *,
    eps: float | None = None,
    dtype: DTypeLike | None = None,
) -> Array1D[np.floating[Any]]:
    r"""Compute a per-node weighted clustering coefficient from [Barrat2004]_.

    .. math::

        c_i = \frac{1}{s_i (d_i - 1)} \sum_{j, k}^n
            \frac{1}{2} (W_{ij} + W_{ik}) A_{ij} A_{ik} A_{jk}

    Note that this clustering coefficient is officially defined for matrices with
    positive weights.

    .. [Barrat2004] A. Barrat, M. Barthélemy, R. Pastor-Satorras, A. Vespignani,
        *The Architecture of Complex Weighted Networks*,
        Proceedings of the National Academy of Sciences, Vol. 101, pp. 3747--3752, 2004,
        `doi:10.1073/pnas.0400087101 <https://doi.org/10.1073/pnas.0400087101>`__.
    """
    n, m = mat.shape
    if n != m:
        raise ValueError(f"matrix not square: {mat.shape}")

    if eps is None:
        try:
            eps = np.sqrt(np.finfo(mat.dtype).eps)
        except ValueError:
            eps = 1.0e-8

    if eps <= 0.0:
        raise ValueError(f"'eps' must be positive: {eps}")

    if __debug__ and np.any(mat < 0):
        raise ValueError("weight matrix 'mat' has non-positive entries")

    A = (np.abs(mat) > eps).astype(dtype)
    strength = compute_weighted_degree(mat)
    degree = np.sum(A, axis=1)

    result = np.sum((mat * A) * (A @ A), axis=1)

    mask = (degree >= 2) & (np.abs(strength) >= eps)
    wcc = np.zeros(n, dtype=dtype)
    wcc[mask] = result[mask] / (strength[mask] * (degree[mask] - 1))

    return wcc


# }}}


# {{{ compute_disparity_serrano


def compute_disparity_serrano(
    mat: Array2D[np.floating[Any]],
    *,
    eps: float | None = None,
    dtype: DTypeLike | None = None,
) -> Array1D[np.floating[Any]]:
    r"""Compute a per-node disparity measure from [Serrano2009]_.

    .. math::

        Y_i = \frac{1}{s_i^2} \sum_{j}^n W_{ij}^2,

    where :math:`s_i` is the weighted degree (see :func:`compute_weighted_degree`).
    This measure is similar to the Inverse Participation Ratio.

    Note that this method is officially defined on positive weight matrices. If
    used for more general weight matrices, the user can take the absolute value.

    .. [Serrano2009] M. Á. Serrano, M. Boguñá, A. Vespignani,
        *Extracting the Multiscale Backbone of Complex Weighted Networks*,
        Proceedings of the National Academy of Sciences, Vol. 106, pp. 6483--6488, 2009,
        `doi:10.1073/pnas.0808904106 <https://doi.org/10.1073/pnas.0808904106>`__.
    """
    n, m = mat.shape
    if n != m:
        raise ValueError(f"matrix not square: {mat.shape}")

    if eps is None:
        try:
            eps = np.sqrt(np.finfo(mat.dtype).eps)
        except ValueError:
            eps = 1.0e-8

    if eps <= 0.0:
        raise ValueError(f"'eps' must be positive: {eps}")

    if __debug__ and np.any(mat < 0):
        raise ValueError("weight matrix 'mat' has non-positive entries")

    strength = compute_weighted_degree(mat)
    mask = strength < eps
    strength[mask] = 1.0

    disparity = np.sum(mat**2, axis=1, dtype=dtype) / strength**2
    disparity[mask] = 0.0

    return disparity


# }}}


# {{{ compute_graph_density


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


# }}}


# {{{ compute_graph_triangles


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


# }}}
