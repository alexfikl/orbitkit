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


# {{{ test_compute_weighted_clustering_coefficient_barrat


def test_compute_weighted_clustering_coefficient_barrat() -> None:
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


# {{{ test_compute_graph_disparity_serrano


def test_compute_graph_disparity_serrano() -> None:
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


# {{{ test_compute_weighted_clustering_coefficient_costantini


def test_compute_weighted_clustering_coefficient_costantini() -> None:
    """Test the values from [Costantini2014]_ Figure 1."""

    from orbitkit.adjacency import make_adjacency_from_edges as make
    from orbitkit.metrics import compute_weighted_clustering_coefficient_costantini

    eps = 1.0e-6
    mats = [
        # Col 1: legs ~0, closing edge = -1
        make(3, {(0, 1): eps, (0, 2): eps, (1, 2): -1.0}, symmetrize=True),
        # Col 2: legs = 1, closing edge ~ -0
        make(3, {(0, 1): 1.0, (0, 2): 1.0, (1, 2): -eps}, symmetrize=True),
        # Col 3: one leg = 1, other leg ~0, closing edge = -1
        make(3, {(0, 1): 1.0, (0, 2): eps, (1, 2): -1.0}, symmetrize=True),
        # Col 4: triangle legs = 1, isolated leg ~0, closing edge = -1
        make(4, {(0, 1): 1.0, (0, 2): 1.0, (0, 3): eps, (1, 2): -1.0}, symmetrize=True),
        # Col 5: triangle legs = 1, isolated leg = 1, closing edge ~ -0
        make(4, {(0, 1): 1.0, (0, 2): 1.0, (0, 3): 1.0, (1, 2): -eps}, symmetrize=True),
        # Col 6: triangle legs ~0, isolated leg ~0, closing edge = -1
        make(4, {(0, 1): eps, (0, 2): eps, (0, 3): eps, (1, 2): -1.0}, symmetrize=True),
        # Col 7: triangle legs = 1, isolated leg ~0, closing edge ~ -0
        make(4, {(0, 1): 1.0, (0, 2): 1.0, (0, 3): eps, (1, 2): -eps}, symmetrize=True),
    ]
    expected_wccs = [
        (-1.0, 0.0, -1.0),  # col 1
        (-1.0, 0.0, 0.0),  # col 2
        (-1.0, 0.0, -1.0),  # col 3
        (-1 / 3, -1 / 3, -1.0),  # col 4
        (-1 / 3, 0.0, 0.0),  # col 5
        (-1 / 3, 0.0, -1 / 3),  # col 6
        (-1 / 3, 0.0, 0.0),  # col 7
    ]
    log.info("")

    for mat, wccs in zip(mats, expected_wccs, strict=True):
        C_W = compute_weighted_clustering_coefficient_costantini(mat, variant=6)
        C_O = compute_weighted_clustering_coefficient_costantini(mat, variant=7)
        C_Z = compute_weighted_clustering_coefficient_costantini(mat, variant=8)

        error_W = np.abs(wccs[0] - C_W[0])
        error_O = np.abs(wccs[1] - C_O[0])
        error_Z = np.abs(wccs[2] - C_Z[0])
        log.info("C_W %.6e C_O %.6e C_Z %.6e", C_W[0], C_O[0], C_Z[0])
        log.info("     %.6e      %.6e      %.6e", error_W, error_O, error_Z)
        assert error_W < eps
        assert error_O < 2.0e-2
        assert error_Z < 2 * eps


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
