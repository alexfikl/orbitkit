# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.linalg as la

from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


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
    graph = Graph.Weighted_Adjacency(w)
    part = leidenalg.find_partition(
        graph,
        # NOTE: this algorithm is used in [Arnulfo2020]
        #   https://doi.org/10.1038/s41467-020-18975-8
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=gamma,
    )

    return tuple(np.array(p) for p in part)


def find_clusters_from_timeseries(
    x: Array,
    *,
    window_length: int | None = None,
    gamma: float = 1.0,
    eps: float = 1.0e-8,
    p: Any = None,
) -> tuple[Array, ...]:
    # {{{ validate inputs

    _, n = x.shape
    if window_length is None:
        window_length = n // 8

    if window_length <= 0:
        raise ValueError(f"'window_length' should be positive: {window_length}")

    # }}}

    # compute matrix
    xw = x[:, -window_length:]
    mat = la.norm(xw[:, None, :] - xw[None, :, :], axis=2, ord=p)

    # zap out very small connections
    mat[np.abs(mat) < eps] = 0.0

    return find_clusters_from_weights(mat, gamma=gamma)
