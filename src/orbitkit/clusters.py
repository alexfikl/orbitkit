# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import cast

import numpy as np

from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


def make_mse_weight_matrix(x: Array, *, alpha: float = 1.0) -> Array:
    r"""Compute a weight matrix based on the pairwise :math:`\ell_2` errors.

    .. math::

        W_{ij} = \exp \left(-\alpha
            \sqrt{\frac{1}{n} \sum_{k = 0}^n (x_{ik} - x_{jk})^2}
        \right)

    These weights will be close to zero when the error is large. Due to the use
    of :math:`\ell_2` errors, only signals that are fully synchronized (both in
    ampliture and phase) will be clustered together.
    """
    # compute error matrix
    mse = np.sqrt(np.mean((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
    mat = np.exp(-alpha * mse)

    # make sure matrix is symmetric
    mat = (mat + mat.T) / 2.0
    np.fill_diagonal(mat, 0.0)

    return cast("Array", mat)


def make_spearman_weight_matrix(x: Array) -> Array:
    r"""Compute a weight matrix based on the Spearman rank coefficient.

    .. math::

        W_{ij} = \mathrm{Sym} max(0, r_{ij})

    where :math:`r_{ij}` are the rank correlation coefficients. This weight matrix
    will only take into account phase synchronized signal and ignore any amplitude
    differences.
    """

    from scipy.stats import spearmanr

    # compute correlation
    corr = spearmanr(x, axis=1)

    # cut out all the anti-correlated variables
    # mat = np.clip(corr.statistic, 0.0, np.inf)
    mat = np.clip(corr.statistic, 0.0, 1.0)

    # make sure matrix is symmetric
    mat = (mat + mat.T) / 2.0
    np.fill_diagonal(mat, 0.0)

    return cast("Array", mat)


def make_plv_weight_matrix(x: Array) -> Array:
    """Compute a weight matrix using the Phase Locking Value (PLV).

    Like :func:`make_spearman_weight_matrix`, this will also only take into
    account phase synchronization.
    """

    from scipy.signal import hilbert

    xhat = hilbert(x, axis=1)
    phases = np.angle(xhat)

    dphi = phases[:, np.newaxis, :] - phases[np.newaxis, :, :]
    plv = np.exp(1j * dphi)

    mat = np.abs(np.mean(plv, axis=2))
    np.fill_diagonal(mat, 0.0)

    return cast("Array", mat)


def find_clusters_from_weights(
    w: Array,
    *,
    gamma: float = 1.0,
) -> tuple[Array, ...]:
    """Determine clusters based on the weight matrix *w*.

    The clusters are determined using the Leiden algorithm. The weight matrix is
    assumed to determine a weighted directed graph on which the algorithm will be
    applied.
    """
    try:
        import leidenalg
    except ImportError:
        raise ImportError("Clustering functionality requires 'leidenalg'") from None

    from igraph import Graph

    # FIXME: igraph also seems to have method to compute communities using the
    # Leiden algorithm, but it seems to need some additional setup (?).
    graph = Graph.Weighted_Adjacency(w, mode="directed", attr="weight")
    part = leidenalg.find_partition(
        graph,
        # NOTE: this algorithm is used in [Arnulfo2020]
        #   https://doi.org/10.1038/s41467-020-18975-8
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=gamma,
        weights="weight",
    )

    return tuple(np.array(p) for p in part)


def find_clusters_from_timeseries(
    x: Array,
    *,
    method: str = "mse",
    window: int | None = None,
    gamma: float = 1.0,
    alpha: float = 1.0,
    normalize: bool = False,
) -> tuple[Array, ...]:
    """Determine synchronized clusters from a given time series *x*.

    :arg method: use one of the predefined weight matrices to determine the
        corresponding graph. This can be one of: "mse", "spearman" or "plv".
    :arg window: the window on which to compute the weight matrix. Setting this
        to 0 will use the full signal.
    :arg gamma: parameter used in the Leiden algorithm (i.e. the resolution
        parameter).
    :arg alpha: parameter used to compute the "mse" weight matrix.
    :arg normalize: if *True*, the z-score of the signal is used to compute the
        weight matrices. This will ignore some amplitude differences between the
        signals.

    :returns: a :class:`tuple` of arrays, each entry denotes a cluster and
        contains the list of variables that are synchronized in that cluster.
    """
    _, n = x.shape
    if window is None:
        window = n // 8

    if window <= 0:
        raise ValueError(f"'window' should be positive: {window}")

    from scipy.stats import zscore

    x = x[:, -window:]
    if normalize:
        # computes the z-score: (x - mu) / sigma
        x = zscore(x, axis=1, ddof=1)

    if method == "mse":
        mat = make_mse_weight_matrix(x, alpha=alpha)
    elif method == "spearman":
        mat = make_spearman_weight_matrix(x)
    elif method == "plv":
        mat = make_plv_weight_matrix(x)
    else:
        raise ValueError(f"Unknown weight matrix computation 'method': {method}")

    return find_clusters_from_weights(mat, gamma=gamma)
