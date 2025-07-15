# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


def make_weight_matrices(
    x: Array,
    *,
    nwindows: int = 1,
    window_length: int | None = None,
    overlap: float = 0.5,
    eps: float = 1.0e-6,
    p: Any = None,
) -> Iterator[Array]:
    from orbitkit.cycles import make_windows

    _, n = x.shape
    if window_length is None:
        window_length = n // 8

    if window_length <= 0:
        raise ValueError(f"'window_length' should be positive: {window_length}")

    from scipy.stats import spearmanr

    for start, end in make_windows(n, nwindows, window_length, overlap=overlap):
        xw = x[:, start:end]
        corr = spearmanr(xw, axis=1)

        # cut out all the anti-correlated variables
        mat = np.clip(corr.statistic, 0.0, np.inf)

        # make sure matrix is symmetric
        mat = (mat + mat.T) / 2.0

        # clip out things that are very close to 0 / 1 (confuses the weighted graph)
        # mat[mat < eps] = 0.0
        # mat[mat > 1.0 - eps] = 1.0
        np.fill_diagonal(mat, 0.0)

        yield mat


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
    window_length: int | None = None,
    gamma: float = 1.0,
    eps: float = 1.0e-6,
    p: Any = None,
) -> tuple[Array, ...]:
    (mat,) = make_weight_matrices(
        x,
        nwindows=1,
        window_length=window_length,
        eps=eps,
        p=p,
    )

    return find_clusters_from_weights(mat, gamma=gamma)
