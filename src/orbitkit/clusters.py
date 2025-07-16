# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import cast

import numpy as np

from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


def make_mse_weight_matrix(x: Array, *, alpha: float = 1.0) -> Array:
    # compute error matrix
    mse = np.sqrt(np.mean((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
    mat = np.exp(-alpha * mse)

    # make sure matrix is symmetric
    mat = (mat + mat.T) / 2.0
    np.fill_diagonal(mat, 0.0)

    return cast("Array", mat)


def make_spearman_weight_matrix(x: Array) -> Array:
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
) -> tuple[Array, ...]:
    _, n = x.shape
    if window is None:
        window = n // 8

    if window <= 0:
        raise ValueError(f"'window' should be positive: {window}")

    from scipy.stats import zscore

    # computes the z-score: (x - mu) / sigma
    x = zscore(x[:, -window:], axis=1, ddof=1)

    if method == "mse":
        mat = make_mse_weight_matrix(x, alpha=alpha)
    elif method == "corr":
        mat = make_spearman_weight_matrix(x)
    elif method == "plv":
        mat = make_plv_weight_matrix(x)
    else:
        raise ValueError(f"Unknown weight matrix computation 'method': {method}")

    return find_clusters_from_weights(mat, gamma=gamma)
