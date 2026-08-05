# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

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
    not mean that the node is isolated. This is also known as the "net degree".
    """

    n, m = mat.shape
    if n != m:
        raise ValueError(f"matrix not square: {mat.shape}")

    return np.sum(mat, axis=1)


def compute_positive_weighted_degree(
    mat: Array2D[np.floating[Any]],
) -> Array1D[np.floating[Any]]:
    r"""Compute the weighted degree of :math:`W^+_{ij} = \max(W_{ij}, 0)`."""
    mat = np.where(mat > 0, mat, 0.0)
    return compute_weighted_degree(mat)


def compute_negative_weighted_degree(
    mat: Array2D[np.floating[Any]],
) -> Array1D[np.floating[Any]]:
    r"""Compute the weighted degree of :math:`W^-_{ij} = \max(-W_{ij}, 0)`."""
    mat = np.where(mat < 0, -mat, 0.0)
    return compute_weighted_degree(mat)


def compute_total_weighted_degree(
    mat: Array2D[np.floating[Any]],
) -> Array1D[np.floating[Any]]:
    r"""Compute the weighted degree of the absolute value :math:`W_{ij} = |W_{ij}|`."""
    return compute_weighted_degree(np.abs(mat))


def compute_normalized_weighted_degree(
    mat: Array2D[np.floating[Any]],
    *,
    eps: float | None = None,
) -> Array1D[np.floating[Any]]:
    r"""Compute a normalized weighted degree.

    .. math::

        D_i = \frac{d_i^+ - d_i^-}{d_i^+ + d_i^-}
            = \frac{\text{net degree}}{\text{total degree}}.
    """

    if eps is None:
        try:
            eps = np.sqrt(np.finfo(mat.dtype).eps)
        except ValueError:
            eps = 1.0e-8

    if eps <= 0.0:
        raise ValueError(f"'eps' must be positive: {eps}")

    net_d = compute_weighted_degree(mat)
    tot_d = compute_total_weighted_degree(mat)

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot_d < eps, 0.0, net_d / tot_d)


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
    positive weights. We also assume that the weight matrix *mat* is symmetric
    and has zero diagonal.

    .. [Barrat2004] A. Barrat, M. Barthélemy, R. Pastor-Satorras, A. Vespignani,
        *The Architecture of Complex Weighted Networks*,
        Proceedings of the National Academy of Sciences, Vol. 101, pp. 3747--3752, 2004,
        `doi:10.1073/pnas.0400087101 <https://doi.org/10.1073/pnas.0400087101>`__.

    :arg eps: tolerance used to clip values close to zero in the matrix
        (when computing the adjacency matrix from *mat*) and for cutting off
        small values of the weighted degree in the formula.
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

    # NOTE: these are heavy, so we don't normally check
    if __debug__:
        if np.any(mat < 0):
            raise ValueError("weight matrix 'mat' has negative entries")

        if np.any(np.abs(np.diag(mat)) > eps):
            raise ValueError("weight matrix 'mat' does not have a zero diagonal")

        if not np.allclose(mat, mat.T, rtol=eps, atol=eps):
            raise ValueError("weight matrix 'mat' is not symmetric")

    A = (np.abs(mat) > eps).astype(dtype)
    strength = compute_weighted_degree(mat)
    degree = np.sum(A, axis=1)

    # NOTE: since W is symmetric, `(W_ij + W_{ik}) / 2 -> W_{ij}` in the sum
    result = np.einsum("ij,ij,ik,kj->i", mat, A, A, A)

    mask = (degree >= 2) & (np.abs(strength) >= eps)
    wcc = np.zeros(n, dtype=dtype)
    wcc[mask] = result[mask] / (strength[mask] * (degree[mask] - 1))

    return wcc


# }}}


# {{{ compute_weighted_clustering_coefficient_costantini


def compute_weighted_clustering_coefficient_costantini(
    mat: Array2D[np.floating[Any]],
    *,
    variant: Literal[6, 7, 8] = 6,
    eps: float | None = None,
    dtype: DTypeLike | None = None,
) -> Array1D[np.floating[Any]]:
    r"""Compute the weighted clustering coefficients from [Costantini2014]_.

    This function implements the 3 generalization of the clustering coefficient
    to signed weighted graphs given in Equation 6, 7, and 8. The precise coefficient
    can be chosen with the *variant* keyword (matching the equation number).

    All the coefficients assume that: (1) the weight matrix *mat* is symmetric,
    (2) that its diagonal is 0, and (3) that the weight is normalized such that
    :math:`\max(|w_{ij}|) = 1`. If the weights are all positive, then variant
    7 is equivalent to the Onnela et al (2005) definition.

    .. [Costantini2014] G. Costantini, M. Perugini,
        *Generalization of Clustering Coefficients to Signed Correlation Networks*,
        PLoS ONE, Vol. 9, pp. e88669--e88669, 2014,
        `doi:10.1371/journal.pone.0088669 <https://doi.org/10.1371/journal.pone.0088669>`__.

    :arg eps: tolerance used to cut off small values from the matrix. If this is
        not desired, just set it to 0.
    :returns: a local cluster coefficient for each node. If the coefficient is not
        defined for a node (e.g. if it does not have sufficient neighbors), then
        the degree is set to NaN.
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

    # NOTE: these are heavy, so we don't normally check
    if __debug__:
        if (mmax := np.max(np.abs(mat))) > 1 + eps:
            raise ValueError(f"weight matrix is not normalized: max(abs(W)) = {mmax}")

        if np.any(np.abs(np.diag(mat)) > eps):
            raise ValueError("weight matrix 'mat' does not have a zero diagonal")

        if not np.allclose(mat, mat.T, rtol=eps, atol=eps):
            raise ValueError("weight matrix 'mat' is not symmetric")

    W = mat.copy()
    if eps != 0:
        W[np.abs(mat) < eps] = 0.0

    if variant == 6:
        A = np.sign(W)
        A3 = np.einsum("ij,jk,ki->i", A, A, A)

        degree = np.sum(W != 0, axis=1, dtype=dtype)
        max_triangles = degree * (degree - 1)

        with np.errstate(invalid="ignore", divide="ignore"):
            wcc = np.where(max_triangles > 0, A3 / max_triangles, np.nan)
    elif variant == 7:
        A = np.cbrt(W)
        A3 = np.einsum("ij,jk,ki->i", A, A, A)

        degree = np.sum(W != 0, axis=1, dtype=dtype)
        max_triangles = degree * (degree - 1)

        with np.errstate(invalid="ignore", divide="ignore"):
            wcc = np.where(max_triangles > 0, A3 / max_triangles, np.nan)
    elif variant == 8:
        W3 = np.einsum("ij,jk,ki->i", W, W, W)
        denominator = np.sum(np.abs(W), axis=1) ** 2 - np.sum(W**2, axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            wcc = np.where(denominator > 0, W3 / denominator, np.nan)
    else:
        raise ValueError(f"unknown coefficient variant: {variant}")

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
        raise ValueError("weight matrix 'mat' has negative entries")

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


# {{{ compute_participation_coefficient


def compute_nx_community_strengths(
    mat: Array2D[np.floating[Any]],
    communities: Sequence[set[int]],
) -> Array2D[np.floating[Any]]:
    r"""Compute the community strengths for each node.

    For each node :math:`i` and community :math:`c`, this computes a sum over all
    the edges between that node and nodes in the community, i.e.

    .. math::

        \kappa_{ic} = \sum_{j \in c} W_{ij}

    In practice, this is similar to a :func:`compute_weighted_degree` and comes
    with the same caveats for signed networks.

    :arg mat: a weighted adjacency matrix of shape ``(n, n)``, where *n* is the
        number of nodes. The matrix is assumed to be symmetric and have a
        zero diagonal.
    :arg communities: a list of communities defined as ``[{node0, node1, ...}, ...],
        i.e. a list of node sets, one for each community. This is the format
        returned by ``networkx.community.greedy_modularity_communities``.
    """
    if mat.ndim != 2:
        raise ValueError(f"adjacency matrix is not 2 dimensional: {mat.shape}")

    n, m = mat.shape
    if n != m:
        raise ValueError(f"adjacency matrix is not square: {mat.shape}")

    from itertools import product

    node_to_community = np.full(n, -1, dtype=np.int32)
    for c, nodes in enumerate(communities):
        for node in nodes:
            node_to_community[node] = c

    if not np.all(node_to_community >= 0):
        raise ValueError("not all nodes are assigned to communities")

    result = np.zeros((n, len(communities)), dtype=mat.dtype)
    for i, j in product(range(n), repeat=2):
        result[i, node_to_community[j]] += mat[i, j]

    return result


def compute_participation_coefficient(
    mat: Array2D[np.floating[Any]],
    community_strengths: Array2D[np.floating[Any]],
) -> Array1D[np.floating[Any]]:
    r"""Compute a weighted participation coefficient from [Guimera2005]_.

    .. math::

        P_i = 1 - \sum_{j \in c} \left(\frac{\kappa_{ij}}{k_i}\right)^2,

    where :math:`\kappa` is the community strength and :math:`k` is total degree.
    Note that the the participation coefficient is only defined for unsigned
    weighted graphs. This is mainly due to the fact that community algorithms
    only work on unsigned graphs.

    .. [Guimera2005] R. Guimerà, L. A. N. Amaral,
        *Cartography of Complex Networks: Modules and Universal Roles*,
        Journal of Statistical Mechanics: Theory and Experiment, 2005,
        `doi:10.1088/1742-5468/2005/02/p02001 <https://doi.org/10.1088/1742-5468/2005/02/p02001>`__.

    :arg community_strengths: an array of shape ``(nnodes, ncommunities)`` that
        describes the strength (weighted degree) of each node to a set of
        communities. See :func:`compute_nx_community_strengths`.
    """
    if mat.ndim != 2:
        raise ValueError(f"adjacency matrix is not 2 dimensional: {mat.shape}")

    n, m = mat.shape
    if n != m:
        raise ValueError(f"adjacency matrix is not square: {mat.shape}")

    if community_strengths.ndim != 2:
        raise ValueError(f"strength is not 2 dimensional: {community_strengths.shape}")

    if community_strengths.shape[0] != n:
        raise ValueError(
            f"'mat' does not match 'community_strengths': expected {n} nodes "
            f"(got {community_strengths.shape[0]} strengths)"
        )

    kappa = np.sum(community_strengths**2, axis=1)
    degree = np.sum(mat, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(degree > 0, 1.0 - kappa / degree**2, 0.0)

    return p


# }}}


# {{{ compute_modularity


def compute_modularity(
    mat: Array2D[np.floating[Any]],
    communities: Sequence[set[int]],
    *,
    eps: float | None = None,
) -> float:
    r"""Compute the signed modularity from [Gomez2009]_.

    .. math::

        M = \frac{1}{w^+ + w^-} \sum_{i = 0}^n \sum_{j = 0}^n \left[
            W_{ij} - \left(\frac{w_i^+ w_j^+}{w^+} - \frac{w_i^- w_j^-}{w^-}\right)
        \right] \delta(C_i, C_j)

    where :math:`W` is the weight matrix, :math:`w^\pm_i` are the node strengths
    for the positive and negative subnetworks and :math:`\delta(C_i, C_j)`
    is 1 when the nodes :math:`(i, j)` are in the same community and 0 otherwise.

    Note that, by construction, this recovers the unsigned modularity when the
    network has only positive weights. The modularity is also zero when there
    is only one community.

    .. [Gomez2009] S. Gómez, P. Jensen, A. Arenas,
        *Analysis of Community Structure in Networks of Correlated Data*,
        Physical Review E, Vol. 80, pp. 16114--16114, 2009,
        `doi:10.1103/physreve.80.016114 <https://doi.org/10.1103/physreve.80.016114>`__.
    """

    n, m = mat.shape
    if n != m:
        raise ValueError(f"matrix not square: {mat.shape}")

    if not communities:
        raise ValueError("'communities' cannot be an empty sequence")

    if eps is None:
        try:
            eps = np.sqrt(np.finfo(mat.dtype).eps)
        except ValueError:
            eps = 1.0e-8

    if eps <= 0.0:
        raise ValueError(f"'eps' must be positive: {eps}")

    if __debug__:  # ruff: ignore[collapsible-if]
        if not np.allclose(mat, mat.T, rtol=eps, atol=eps):
            raise ValueError("'mat' is not a symmetric weight matrix")

    w_pos = compute_positive_weighted_degree(mat)
    from orbitkit.clusters import community_labels

    labels = community_labels(communities, n)
    if np.any(labels < 0):
        raise ValueError("not all nodes are in 'communities'")

    w_neg = compute_negative_weighted_degree(mat)

    W_pos = np.sum(w_pos)
    W_neg = np.sum(w_neg)

    total_W = W_pos + W_neg
    if total_W < eps:
        return 0.0

    with np.errstate(invalid="ignore", divide="ignore"):
        w_pos /= np.sqrt(W_pos) if W_pos > eps else 1.0
        w_neg /= np.sqrt(W_neg) if W_neg > eps else 1.0

    W = mat - (np.outer(w_pos, w_pos) - np.outer(w_neg, w_neg))
    C = labels[:, None] == labels[None, :]

    return np.sum(W * C) / total_W


# }}}


# {{{ compute_eigenvector_centrality


class EigenvectorCentrality(NamedTuple):
    lambda_max: float
    """The largest real eigenvalue."""
    score: Array1D[np.floating[Any]]
    """An unsigned eigenvector centrality score."""
    eigenbasis: Array2D[np.floating[Any]]
    """The corresponding basis for the eigenspace spanned by the eigenvalue
    :attr:`lambda_max`. If the eigenvalue is simple, then this will just be a
    single vector that carries sign information (unlike :attr:`score`)
    """


def compute_eigenvector_centrality(
    mat: Array2D[np.floating[Any]],
    *,
    eps: float | None = None,
) -> EigenvectorCentrality:
    r"""Compute a signed version of eigenvector centrality based on [Bonacich2004]_.

    We solve the standard eigenproblem :math:`W v = \lambda v`. Unlike the
    unsigned case, [Bonacich2004]_ picks the largest eigenvalue directly, not the
    eigenvalue with the largest magnitude.

    It is also not guaranteed that the largest eigenvalue is simple, so we
    return the whole eigenspace for higher multiplicity.

    .. [Bonacich2004] P. Bonacich, P. Lloyd,
        *Calculating Status With Negative Relations*,
        Social Networks, Vol. 26, pp. 331--338, 2004,
        `doi:10.1016/j.socnet.2004.08.007 <https://doi.org/10.1016/j.socnet.2004.08.007>`__.
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

    if __debug__:  # ruff: ignore[collapsible-if]
        if not np.allclose(mat, mat.T, rtol=eps, atol=eps):
            raise ValueError("'mat' is not a symmetric weight matrix")

    # gen the eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(mat)

    # find multiplicity
    eps *= max(np.abs(eigvals[0]), np.abs(eigvals[-1]), 1.0)
    mask = eigvals > (eigvals[-1] - eps)
    k = np.sum(mask)

    # find eigenspace
    V = eigvecs[:, mask]
    score = np.einsum("ij,ij->i", V, V)

    # NOTE: both v and -v are eigenvectors for the eigenvalue. We fix
    # the sign so that the largest entry in v is positive, for consistency
    if k == 1 and V[np.argmax(np.abs(V)), 0] < 0:
        V = -V

    return EigenvectorCentrality(lambda_max=eigvals[-1], score=score, eigenbasis=V)


# }}}


# {{{ compute_assortativity_li


def compute_assortativity_li(
    mat: Array2D[np.floating[Any]],
    *,
    variant: Literal[2, 3, 4, 5, 6, 7] = 2,
    eps: float | None = None,
) -> float:
    """Computes the assortativity measures from [Li2020]_.

    All the measures assume that: (1) the weight matrix *mat* is symmetric,
    (2) that it's diagonal is zero, and (3) that it is undirected.

    .. [Li2020] A.-W. Li, J. Xiao, X.-K. Xu,
        *The Family of Assortativity Coefficients in Signed Social Networks*,
        IEEE Transactions on Computational Social Systems, Vol. 7, pp. 1460--1468, 2020,
        `doi:10.1109/tcss.2020.3023729 <https://doi.org/10.1109/tcss.2020.3023729>`__.

    :arg variant: one of the 6 variants of assortativity from the paper. They are
        named after the equation numbers.
    :returns: assortativity coefficient in :math:`[-1, 1]`. It can also return
        *NaN* if the weight variance is zero (i.e. every edge has the same weight).
    """
    n, m = mat.shape
    if n != m:
        raise ValueError(f"matrix not square: {mat.shape}")

    if n == 0:
        raise ValueError(f"assortativity not defined for empty matrices: {mat}")

    if eps is None:
        try:
            eps = np.sqrt(np.finfo(mat.dtype).eps)
        except ValueError:
            eps = 1.0e-8

    if eps <= 0.0:
        raise ValueError(f"'eps' must be positive: {eps}")

    if not 2 <= variant <= 7:
        raise ValueError(f"unknown 'variant': {variant!r} (not in 2-7)")

    if __debug__:
        if np.any(np.abs(np.diag(mat)) > eps):
            raise ValueError("weight matrix 'mat' does not have a zero diagonal")

        if not np.allclose(mat, mat.T, rtol=eps, atol=eps):
            raise ValueError("weight matrix 'mat' is not symmetric")

    if variant in {2, 3, 4, 6, 7} and np.max(mat) < eps:
        raise ValueError("there are no edges with positive weights in 'mat'")

    if variant in {3, 4, 5, 6, 7} and np.min(mat) > -eps:
        raise ValueError("there are no edges with negative weights in 'mat'")

    if variant == 2:  # r+(+, +)
        A_pos = (mat > eps).astype(mat.dtype)

        j_pos = np.sum(A_pos, axis=1)
        i, j = np.where(np.triu(A_pos, k=1))

        a, b = j_pos[i], j_pos[j]
    elif variant == 3:  # r-(+, +)
        A_pos = (mat > eps).astype(mat.dtype)
        A_neg = (mat < -eps).astype(mat.dtype)

        j_pos = np.sum(A_pos, axis=1)
        i, j = np.where(np.triu(A_neg, k=1))

        a, b = j_pos[i], j_pos[j]
    elif variant == 4:  # r+(-, -)
        A_pos = (mat > eps).astype(mat.dtype)
        A_neg = (mat < -eps).astype(mat.dtype)

        j_neg = np.sum(A_neg, axis=1)
        i, j = np.where(np.triu(A_pos, k=1))

        a, b = j_neg[i], j_neg[j]
    elif variant == 5:  # r-(-, -)
        A_neg = (mat < -eps).astype(mat.dtype)

        j_neg = np.sum(A_neg, axis=1)
        i, j = np.where(np.triu(A_neg, k=1))

        a, b = j_neg[i], j_neg[j]
    elif variant == 6:  # r+(+, -)
        A_pos = (mat > eps).astype(mat.dtype)
        A_neg = (mat < -eps).astype(mat.dtype)

        j_pos = np.sum(A_pos, axis=1)
        j_neg = np.sum(A_neg, axis=1)
        i, j = np.where(np.triu(A_pos, k=1))

        # NOTE: sample both directions to not depend on node indexing
        a = np.concatenate([j_pos[i], j_pos[j]])
        b = np.concatenate([j_neg[j], j_neg[i]])
    elif variant == 7:  # r-(+, -)
        A_pos = (mat > eps).astype(mat.dtype)
        A_neg = (mat < -eps).astype(mat.dtype)

        j_pos = np.sum(A_pos, axis=1)
        j_neg = np.sum(A_neg, axis=1)
        i, j = np.where(np.triu(A_neg, k=1))

        a = np.concatenate([j_pos[i], j_pos[j]])
        b = np.concatenate([j_neg[j], j_neg[i]])
    else:
        raise AssertionError

    term1 = np.sum(a * b) / a.size
    term2 = (0.5 * np.sum(a + b) / a.size) ** 2
    term3 = 0.5 * np.sum(a**2 + b**2) / a.size

    # NOTE: return nan rather than returning infinity to keep the assortativity in
    # [-1, 1] for all valid inputs.
    if term3 - term2 == 0:
        return np.nan

    return (term1 - term2) / (term3 - term2)


# }}}


# {{{ compute_assortativity_arcagni


def compute_assortativity_arcagni(
    mat: Array2D[np.floating[Any]],
    *,
    eps: float | None = None,
) -> float:
    r"""Computes the assortativity measure from [Arcagni2021]_.

    .. math::

        \rho(\mathbf{s}, \mathbf{E})
        = \frac{\mathbf{s}^T (\mathbf{E} - \mathbf{q} \mathbf{q}^T) \mathbf{s}}
               {\mathbf{s}^T (\mathbf{D}_q - \mathbf{q} \mathbf{q}^T) \mathbf{s}}
        = \frac{\mathbf{s}^T \mathbf{E} \mathbf{s} - (\mathbf{s}^T \mathbf{q})^2}
            {\mathbf{s}^T (\mathbf{q} \odot \mathbf{s}) - (\mathbf{s}^T \mathbf{q})^2},

    where :math:`\mathbf{s}` is the node strength, :math:`\mathbf{E} = \mathbf{W}
    / \omega` is the normalized weight matrix, :math:`\mathbf{q} = \mathbf{E} 1`,
    and :math:`\mathbf{D}_q` is a diagonal matrix with :math:`\mathbf{q}` on the
    diagonal.

    All the measures assume that: (1) the weight matrix *mat* is symmetric,
    (2) that it's diagonal is zero, and (3) that the weights are positive.

    .. [Arcagni2021] A. Arcagni, R. Grassi, S. Stefani, A. Torriero,
        *Extending Assortativity: An Application to Weighted Social Networks*,
        Journal of Business Research, Vol. 129, pp. 774--783, 2021,
        `doi:10.1016/j.jbusres.2019.10.008 <https://doi.org/10.1016/j.jbusres.2019.10.008>`__.

    :returns: assortativity coefficient in :math:`[-1, 1]`. It can also return
        *NaN* if assortativity is not defined.
    """
    n, m = mat.shape
    if n != m:
        raise ValueError(f"matrix not square: {mat.shape}")

    if n == 0:
        raise ValueError(f"assortativity not defined for empty matrices: {mat}")

    if eps is None:
        try:
            eps = np.sqrt(np.finfo(mat.dtype).eps)
        except ValueError:
            eps = 1.0e-8

    if eps <= 0.0:
        raise ValueError(f"'eps' must be positive: {eps}")

    if __debug__:
        if np.any(np.abs(np.diag(mat)) > eps):
            raise ValueError("weight matrix 'mat' does not have a zero diagonal")

        if np.any(mat < 0):
            raise ValueError("weight matrix 'mat' has negative entries")

        if not np.allclose(mat, mat.T, rtol=eps, atol=eps):
            raise ValueError("weight matrix 'mat' is not symmetric")

    omega = np.sum(mat)
    if omega == 0:
        return np.nan

    E = mat / omega
    q = np.sum(E, axis=1)
    c = np.sum(mat, axis=1)

    cq = (c @ q) ** 2
    denominator = c @ (c * q) - cq
    if denominator == 0:
        return np.nan

    numerator = c @ (E @ c) - cq
    return numerator / denominator


# }}}


# {{{ compute_local_assortativity_sabek


def compute_local_assortativity_sabek(
    mat: Array2D[np.floating[Any]],
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float | None = None,
) -> Array1D[np.floating[Any]]:
    r"""Compute the weighted local assortativity from [Sabek2023]_.

    .. math::

        \rho_i(\alpha, \beta) =
            \frac{\omega_e^\beta [l_e - U(\alpha, \beta)] [m_e - U(\alpha, \beta)]}
                 {\Omega \sigma^2(\alpha, \beta)},

    where the parameters are described in Equation (18) from [Sabek2023]_.
    Furthermore, the associated global assortativity can be directly obtained as

    .. math::

        r(\alpha, \beta) = \frac{1}{2} \sum \rho_i(\alpha, \beta).

    Note that the parameters :math:`(\alpha, \beta)` are allowed to vary in
    :math:`[0, 1]`. However, only the boundary values are easily interpretable
    (e.g. degree vs strength used in formulas). See [Sabek2023]_ for details.

    The formula assumes that: (1) the weight matrix *mat* is symmetric,
    (2) that it's diagonal is zero, and (3) that the weights are positive.

    .. [Sabek2023] M. Sabek, U. Pigorsch,
        *Local Assortativity in Weighted and Directed Complex Networks*,
        Physica A: Statistical Mechanics and Its Applications, Vol. 630,
        pp. 129231--129231, 2023,
        `doi:10.1016/j.physa.2023.129231 <https://doi.org/10.1016/j.physa.2023.129231>`__.

    :arg alpha: parameter in :math:`[0, 1]` used in the formula from [Sabek2023]_.
        A value of 0 uses the degree (unweighted), while a value of 1 uses the
        strength (weighted degree).
    :arg beta: parameter in :math:`[0, 1]` used in the formula from [Sabek2023]_.
        A value of 0 uses an unweighted correlation, while a value of 1 uses a
        fully weighted correlation.
    :arg returns: an array of shape ``(n,)`` per node in the network. If the
        local assortativity cannot be computed (e.g. zero variance), then the
        array will be *NaN*.
    """

    n, m = mat.shape
    if n != m:
        raise ValueError(f"matrix not square: {mat.shape}")

    if n == 0:
        raise ValueError(f"assortativity not defined for empty matrices: {mat}")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"'alpha' must be in [0, 1]: {alpha!r}")

    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"'beta' must be in [0, 1]: {beta!r}")

    if eps is None:
        try:
            eps = np.sqrt(np.finfo(mat.dtype).eps)
        except ValueError:
            eps = 1.0e-8

    if eps <= 0.0:
        raise ValueError(f"'eps' must be positive: {eps}")

    if __debug__:
        if np.any(np.abs(np.diag(mat)) > eps):
            raise ValueError("weight matrix 'mat' does not have a zero diagonal")

        if np.any(mat < 0):
            raise ValueError("weight matrix 'mat' has negative entries")

        if not np.allclose(mat, mat.T, rtol=eps, atol=eps):
            raise ValueError("weight matrix 'mat' is not symmetric")

    # get upper triangular elements
    iu, ju = np.triu_indices(n, k=1)
    w = mat[iu, ju]

    # mask out zero elements
    mask = w > 0
    i, j, w = iu[mask], ju[mask], w[mask]

    # compute degree strength
    w_alpha = w**alpha
    w_beta = w**beta
    s_star = np.bincount(i, w_alpha, minlength=n) + np.bincount(j, w_alpha, minlength=n)

    # compute variables
    l_e = s_star[i] - w_alpha
    m_e = s_star[j] - w_alpha

    Omega = np.sum(w_beta)
    if Omega == 0:
        return np.full(n, np.nan, dtype=mat.dtype)

    U = np.sum(w_beta * (l_e + m_e)) / (2 * Omega)
    sigma_sqr = np.sum(w_beta * (l_e**2 + m_e**2)) / (2 * Omega) - U**2
    if sigma_sqr <= 0:
        return np.full(n, np.nan, dtype=mat.dtype)

    # compute local assortativity
    rho_e = w_beta * (l_e - U) * (m_e - U) / (Omega * sigma_sqr)
    rho_v = np.bincount(i, rho_e, minlength=n) + np.bincount(j, rho_e, minlength=n)

    return rho_v  # ty: ignore[invalid-return-type]


# }}}
