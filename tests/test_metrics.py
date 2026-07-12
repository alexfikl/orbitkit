# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from orbitkit.utils import module_logger

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)

# {{{ test_compute_weighted_degree


def test_compute_weighted_degree() -> None:
    from orbitkit.metrics import compute_weighted_degree

    with pytest.raises(ValueError, match="not square"):
        compute_weighted_degree(np.ones((3, 4)))

    W = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    s = compute_weighted_degree(W)
    assert s.shape == (3,)
    assert np.allclose(s, [3.0, 4.0, 5.0])


# }}}


# {{{ test_compute_weighted_clustering_coefficient


def test_compute_weighted_clustering_coefficient() -> None:
    from orbitkit.metrics import compute_weighted_clustering_coefficient_barrat

    with pytest.raises(ValueError, match="not square"):
        compute_weighted_clustering_coefficient_barrat(np.ones((3, 4)))

    with pytest.raises(ValueError, match="'eps' must be positive"):
        compute_weighted_clustering_coefficient_barrat(np.eye(3), eps=-1.0)

    # isolated nodes (no edges) → coefficient is 0 for all nodes
    n = 5
    W = np.zeros((n, n))
    wcc = compute_weighted_clustering_coefficient_barrat(W)
    assert wcc.shape == (n,)
    assert np.allclose(wcc, 0.0)

    W = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
    wcc = compute_weighted_clustering_coefficient_barrat(W)
    assert np.allclose(wcc, 0.0)

    # 3-node complete graph with uniform weights w.
    w = 3.0
    W = w * (np.ones((3, 3)) - np.eye(3))
    wcc = compute_weighted_clustering_coefficient_barrat(W)
    assert np.allclose(wcc, 1.0, atol=1.0e-12)


# }}}


# {{{ test_compute_graph_disparity


def test_compute_graph_disparity() -> None:
    from orbitkit.metrics import compute_disparity_serrano

    with pytest.raises(ValueError, match="not square"):
        compute_disparity_serrano(np.ones((3, 4)))

    with pytest.raises(ValueError, match="'eps' must be positive"):
        compute_disparity_serrano(np.eye(3), eps=-1.0)

    # isolated nodes
    n = 4
    W = np.zeros((n, n))
    Y = compute_disparity_serrano(W)
    assert Y.shape == (n,)
    assert np.allclose(Y, 0.0)

    # single-edge node: all weight on one neighbour
    W = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    Y = compute_disparity_serrano(W)
    assert np.allclose(Y[0], 1.0, atol=1.0e-10)

    # uniform weights: disparity = 1/k for a node with k equal-weight edges
    n = 5
    W = np.ones((n, n)) - np.eye(n)  # k = n-1 = 4 equal neighbours
    Y = compute_disparity_serrano(W)
    assert np.allclose(Y, 1.0 / (n - 1), atol=1.0e-10)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
