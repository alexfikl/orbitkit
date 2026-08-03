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

# {{{ test_leiden_communities_signed_triangle


def test_leiden_communities_signed_triangle() -> None:
    from orbitkit.clusters import signed_leiden_communities

    mat = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, -1.0, 0.0],
    ])

    communities = signed_leiden_communities(mat, seed=42)
    membership = {node: i for i, comm in enumerate(communities) for node in comm}

    assert len(communities) == 2
    assert membership[0] == membership[1]
    assert membership[0] != membership[2]


# }}}

# {{{ test_leiden_communities_signed_two_cliques


def test_leiden_communities_signed_two_cliques() -> None:
    """Two positive cliques separated by negative edges."""
    from orbitkit.clusters import signed_leiden_communities

    mat = np.array([
        [0.0, 1.0, 1.0, -1.0, -1.0, -1.0],
        [1.0, 0.0, 1.0, -1.0, -1.0, -1.0],
        [1.0, 1.0, 0.0, -1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0, 0.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0, 1.0, 0.0, 1.0],
        [-1.0, -1.0, -1.0, 1.0, 1.0, 0.0],
    ])

    communities = signed_leiden_communities(mat, seed=42)
    membership = {node: i for i, comm in enumerate(communities) for node in comm}

    assert len(communities) == 2
    for node in range(3):
        assert membership[node] == membership[0]
    for node in range(3, 6):
        assert membership[node] == membership[3]
    assert membership[0] != membership[3]


# }}}

# {{{ test_leiden_communities_signed_resolution


def test_leiden_communities_signed_resolution() -> None:
    """Resolution parameter affects community granularity."""
    from orbitkit.clusters import signed_leiden_communities

    mat = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.1, 0.0],
        [0.0, 0.1, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    n_low = len(signed_leiden_communities(mat, resolution=0.01, seed=42))
    n_high = len(signed_leiden_communities(mat, resolution=100.0, seed=42))

    assert n_low <= n_high


# }}}

# {{{ test_leiden_communities_signed_reproducible


def test_leiden_communities_signed_reproducible() -> None:
    """Same seed gives the same partition."""
    from orbitkit.clusters import signed_leiden_communities

    rng = np.random.default_rng(42)
    mat = rng.normal(0, 0.5, (10, 10))
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 0.0)

    communities1 = signed_leiden_communities(mat, seed=42)
    communities2 = signed_leiden_communities(mat, seed=42)

    assert communities1 == communities2


# }}}

# {{{ test_leiden_communities_signed_edge_cases


def test_leiden_communities_signed_edge_cases() -> None:
    from orbitkit.clusters import signed_leiden_communities

    # single node
    communities = signed_leiden_communities(np.zeros((1, 1)), seed=42)
    assert len(communities) == 1

    # all zero: each node is its own community (no edges to bind nodes)
    communities = signed_leiden_communities(np.zeros((4, 4)), seed=42)
    assert len(communities) == 4

    # all positive: single community
    mat = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    communities = signed_leiden_communities(mat, seed=42)
    assert len(communities) == 1


# }}}

# {{{ test_leiden_communities_signed_validation


def test_leiden_communities_signed_validation() -> None:
    from orbitkit.clusters import signed_leiden_communities

    with pytest.raises(ValueError, match="not square"):
        signed_leiden_communities(np.ones((3, 4)))

    with pytest.raises(ValueError, match="non-finite"):
        signed_leiden_communities(np.array([[np.nan, 0], [0, np.nan]]))

    with pytest.raises(ValueError, match="non-finite"):
        signed_leiden_communities(np.array([[np.inf, 0], [0, np.inf]]))


# }}}


if __name__ == "__main__":
    pytest.main([__file__])
