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

    # isolated nodes (no edges) -> coefficient is 0 for all nodes
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


# {{{ test_compute_nx_community_strengths


def test_compute_nx_community_strengths() -> None:
    from orbitkit.metrics import compute_nx_community_strengths

    # non-square
    with pytest.raises(ValueError, match="not square"):
        compute_nx_community_strengths(np.ones((3, 4)), [{0, 1}, {2}])

    # incomplete partition
    n = 4
    mat = np.ones((n, n)) - np.eye(n)
    with pytest.raises(ValueError, match="not all nodes are assigned"):
        compute_nx_community_strengths(mat, [{0, 1}])

    # two communities on a complete 4-node graph (w=1)
    mat = np.ones((n, n)) - np.eye(n)
    communities = [{0, 1}, {2, 3}]
    strengths = compute_nx_community_strengths(mat, communities)
    assert strengths.shape == (n, 2)
    # node 0: 1 edge within community {0,1}, 2 edges to community {2,3}
    assert np.allclose(strengths[0], [1.0, 2.0])
    assert np.allclose(strengths[1], [1.0, 2.0])
    assert np.allclose(strengths[2], [2.0, 1.0])
    assert np.allclose(strengths[3], [2.0, 1.0])


# }}}


# {{{ test_compute_participation_coefficient


def _make_complete(n: int) -> np.ndarray:
    return np.ones((n, n)) - np.eye(n)


def test_compute_participation_coefficient_errors() -> None:
    from orbitkit.metrics import compute_participation_coefficient

    n = 4
    mat = _make_complete(n)
    str_ok = np.zeros((n, 2))

    with pytest.raises(ValueError, match="not square"):
        compute_participation_coefficient(np.ones((3, 4)), str_ok)

    with pytest.raises(ValueError, match="not 2 dimensional"):
        compute_participation_coefficient(mat, np.zeros(n))  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError, match="does not match"):
        compute_participation_coefficient(mat, np.zeros((n + 1, 2)))


def test_compute_participation_coefficient_single_community() -> None:
    from orbitkit.metrics import (
        compute_nx_community_strengths,
        compute_participation_coefficient,
    )

    n = 6
    mat = _make_complete(n)
    communities = [set(range(n))]
    strengths = compute_nx_community_strengths(mat, communities)
    p = compute_participation_coefficient(mat, strengths)
    assert p.shape == (n,)
    assert np.allclose(p, 0.0)


def test_compute_participation_coefficient_balanced() -> None:
    from orbitkit.metrics import (
        compute_nx_community_strengths,
        compute_participation_coefficient,
    )

    # 6 nodes, 3 communities of 2 each, complete graph (w=1)
    # Each node: degree = 5
    #   strength within own community = 1 (one partner)
    #   strength to each other community = 2 (two nodes each)
    # P = 1 - ((1/5)^2 + (2/5)^2 + (2/5)^2) = 1 - 9/25 = 16/25
    n = 6
    mat = _make_complete(n)
    communities = [{0, 1}, {2, 3}, {4, 5}]
    strengths = compute_nx_community_strengths(mat, communities)
    p = compute_participation_coefficient(mat, strengths)
    assert p.shape == (n,)
    assert np.allclose(p, 16.0 / 25.0)


def test_compute_participation_coefficient_isolated_nodes() -> None:
    from orbitkit.metrics import (
        compute_nx_community_strengths,
        compute_participation_coefficient,
    )

    # 2 isolated nodes, 2 connected nodes, each in their own community
    n = 4
    mat = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    )
    communities = [{0}, {1}, {2}, {3}]
    strengths = compute_nx_community_strengths(mat, communities)
    p = compute_participation_coefficient(mat, strengths)
    assert p.shape == (n,)
    assert np.allclose(p[0], 0.0)
    assert np.allclose(p[1], 0.0)
    # nodes 2 and 3 each connect to exactly one other community
    # P = 1 - (1/1)^2 = 0
    assert np.allclose(p[2], 0.0)
    assert np.allclose(p[3], 0.0)


# }}}


# {{{ test_compute_modularity


def test_compute_modularity_single_community() -> None:
    """Signed modularity is zero when all nodes are in one community."""
    from orbitkit.metrics import compute_modularity

    mat = np.array([
        [0.0, 1.0, -1.0],
        [1.0, 0.0, -1.0],
        [-1.0, -1.0, 0.0],
    ])
    communities = ({0, 1, 2},)
    assert compute_modularity(mat, communities) == pytest.approx(0.0)


def test_compute_modularity_unsigned_recovery() -> None:
    """Signed modularity recovers the Newman modularity for positive-only graphs."""
    from orbitkit.metrics import compute_modularity

    mat = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    communities = ({0, 1}, {2})

    result = compute_modularity(mat, communities)
    expected = -2.0 / 9.0
    assert result == pytest.approx(expected)


def test_compute_modularity_signed_known_value() -> None:
    """Hand-calculated signed modularity for a 3-node signed graph."""
    from orbitkit.metrics import compute_modularity

    mat = np.array([
        [0.0, 1.0, -1.0],
        [1.0, 0.0, -1.0],
        [-1.0, -1.0, 0.0],
    ])
    communities = ({0, 1}, {2})

    result = compute_modularity(mat, communities)
    assert result == pytest.approx(1.0 / 3.0)


def test_compute_modularity_all_zero() -> None:
    """Modularity is zero for an all-zero matrix."""
    from orbitkit.metrics import compute_modularity

    mat = np.zeros((4, 4))
    communities = ({0, 1}, {2, 3})
    assert compute_modularity(mat, communities) == pytest.approx(0.0)


def test_compute_modularity_validation() -> None:
    """Input validation for compute_modularity."""
    from orbitkit.metrics import compute_modularity

    with pytest.raises(ValueError, match="not square"):
        compute_modularity(np.ones((3, 4)), ({0, 1},))

    with pytest.raises(ValueError, match="'communities' cannot be an empty sequence"):
        compute_modularity(np.eye(3), ())


# }}}


# {{{ test_compute_eigenvector_centrality


def test_compute_eigenvector_centrality_positive() -> None:
    """Known eigenvector centrality for a 3-node positive chain."""
    from orbitkit.metrics import compute_eigenvector_centrality

    mat = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])

    result = compute_eigenvector_centrality(mat)

    # largest eigenvalue of the chain is sqrt(2)
    assert result.lambda_max == pytest.approx(np.sqrt(2))

    # middle node (1) is the most central
    assert result.score[1] == pytest.approx(0.5)
    assert result.score[0] == pytest.approx(0.25)
    assert result.score[2] == pytest.approx(0.25)

    # all signs in eigenbasis are the same (positive chain)
    assert np.all(result.eigenbasis > 0)


def test_compute_eigenvector_centrality_signed() -> None:
    """Signed centrality: bridge node between positive and negative edges."""
    from orbitkit.metrics import compute_eigenvector_centrality

    mat = np.array([
        [0.0, 2.0, 0.0],
        [2.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ])

    result = compute_eigenvector_centrality(mat)

    # largest eigenvalue is sqrt(5)
    assert result.lambda_max == pytest.approx(np.sqrt(5))

    # node 1 is the bridge - highest centrality
    assert result.score[1] > result.score[0]
    assert result.score[1] > result.score[2]

    # positive edge (0,1) -> same sign; negative edge (1,2) -> opposite sign
    v = result.eigenbasis[:, 0]
    assert np.sign(v[0]) == np.sign(v[1])
    assert np.sign(v[1]) != np.sign(v[2])


def test_compute_eigenvector_centrality_negative() -> None:
    """Purely negative edge gives opposite signs in the eigenbasis."""
    from orbitkit.metrics import compute_eigenvector_centrality

    mat = np.array([
        [0.0, -1.0],
        [-1.0, 0.0],
    ])

    result = compute_eigenvector_centrality(mat)

    assert result.lambda_max == pytest.approx(1.0)
    assert np.allclose(result.score, [0.5, 0.5])
    assert np.sign(result.eigenbasis[0, 0]) != np.sign(result.eigenbasis[1, 0])


def test_compute_eigenvector_centrality_validation() -> None:
    """Input validation for compute_eigenvector_centrality."""
    from orbitkit.metrics import compute_eigenvector_centrality

    with pytest.raises(ValueError, match="not square"):
        compute_eigenvector_centrality(np.ones((3, 4)))

    with pytest.raises(ValueError, match="not a symmetric"):
        compute_eigenvector_centrality(np.array([[0.0, 1.0], [0.0, 0.0]]))


# }}}


# {{{ test_compute_eigenvector_centrality_degenerate


def test_compute_eigenvector_centrality_degenerate() -> None:
    """Largest eigenvalue with multiplicity > 1 across disconnected blocks."""
    from orbitkit.metrics import compute_eigenvector_centrality

    # two disconnected signed 2-node blocks, each with eigenvalues {+1, -1}
    mat = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0, 0.0],
    ])

    result = compute_eigenvector_centrality(mat)

    assert result.lambda_max == pytest.approx(1.0)
    assert result.eigenbasis.shape[1] >= 2
    assert np.allclose(result.score, 0.5)


# }}}


# {{{ test_compute_assortativity_li


def test_compute_assortativity_li_validation() -> None:
    """Input validation for compute_assortativity_li."""
    from orbitkit.metrics import compute_assortativity_li

    # signed matrix with both a positive and a negative edge
    mat = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    with pytest.raises(ValueError, match="not square"):
        compute_assortativity_li(np.ones((3, 4)))

    with pytest.raises(ValueError, match="not defined for empty matrices"):
        compute_assortativity_li(np.zeros((0, 0)))

    with pytest.raises(ValueError, match="'eps' must be positive"):
        compute_assortativity_li(mat, eps=-1.0)

    with pytest.raises(ValueError, match="unknown 'variant'"):
        compute_assortativity_li(mat, variant=1)  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError, match="zero diagonal"):
        compute_assortativity_li(np.array([[1.0, 1.0], [1.0, 0.0]]), variant=2)

    with pytest.raises(ValueError, match="not symmetric"):
        compute_assortativity_li(
            np.array([
                [0.0, 1.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
            ]),
            variant=2,
        )

    # variant 2 requires at least one positive edge
    with pytest.raises(ValueError, match="no edges with positive weights"):
        compute_assortativity_li(-np.abs(mat), variant=2)

    # variant 5 requires at least one negative edge
    with pytest.raises(ValueError, match="no edges with negative weights"):
        compute_assortativity_li(np.abs(mat), variant=5)


def test_compute_assortativity_li_variant2_known_value() -> None:
    """Hand-computed r+(+,+) on a 4-node positive graph."""
    from orbitkit.metrics import compute_assortativity_li

    mat = np.array([
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    assert compute_assortativity_li(mat, variant=2) == pytest.approx(-5.0 / 7.0)
    assert compute_assortativity_li(-mat, variant=5) == pytest.approx(-5.0 / 7.0)


def test_compute_assortativity_li_signed_path() -> None:
    """Hand-computed coefficients for all variants on a signed 4-node graph."""
    from orbitkit.metrics import compute_assortativity_li

    mat = np.array([
        [0.0, 1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0, -1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [0.0, -1.0, 1.0, 0.0],
    ])
    assert compute_assortativity_li(mat, variant=2) == pytest.approx(-0.5)
    assert compute_assortativity_li(mat, variant=3) == pytest.approx(-1.0)
    assert compute_assortativity_li(mat, variant=6) == pytest.approx(-0.5)
    assert compute_assortativity_li(mat, variant=7) == pytest.approx(-1.0 / 3.0)
    assert np.isnan(compute_assortativity_li(mat, variant=4))
    assert np.isnan(compute_assortativity_li(mat, variant=5))


def test_compute_assortativity_li_integer_dtype() -> None:
    """Integer matrices exercise the 'eps' fallback when finfo fails."""
    from orbitkit.metrics import compute_assortativity_li

    mat = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ])
    assert compute_assortativity_li(mat, variant=2) == pytest.approx(-5.0 / 7.0)


def test_compute_assortativity_li_explicit_eps_matches_default() -> None:
    """Passing eps explicitly does not change the result for well-separated weights."""
    from orbitkit.metrics import compute_assortativity_li

    mat = np.array([
        [0.0, 1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0, -1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [0.0, -1.0, 1.0, 0.0],
    ])
    for variant in (2, 3, 6, 7):
        r_default = compute_assortativity_li(mat, variant=variant)
        r_explicit = compute_assortativity_li(mat, variant=variant, eps=1.0e-8)
        assert r_default == pytest.approx(r_explicit)


def test_compute_assortativity_li_permutation_invariance() -> None:
    """Relabeling nodes must not change any of the coefficients."""
    from orbitkit.metrics import compute_assortativity_li

    rng = np.random.default_rng(42)
    mat = np.array([
        [0.0, 1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0, -1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [0.0, -1.0, 1.0, 0.0],
    ])
    perm = rng.permutation(mat.shape[0])
    matp = mat[perm][:, perm]

    for variant in (2, 3, 4, 5, 6, 7):
        r = compute_assortativity_li(mat, variant=variant)
        rp = compute_assortativity_li(matp, variant=variant)
        if np.isnan(r):
            assert np.isnan(rp)
        else:
            assert r == pytest.approx(rp)


def test_compute_assortativity_li_in_range() -> None:
    """Coefficients stay in [-1, 1] (or NaN) on random signed graphs."""
    from orbitkit.metrics import compute_assortativity_li

    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(4, 12))
        mat = rng.uniform(-1.0, 1.0, size=(n, n))
        mat = (mat + mat.T) / 2.0
        np.fill_diagonal(mat, 0.0)
        if np.max(mat) <= 0.0 or np.min(mat) >= 0.0:
            continue

        for variant in (2, 3, 4, 5, 6, 7):
            r = compute_assortativity_li(mat, variant=variant)
            assert np.isnan(r) or -1.0 <= r <= 1.0


# }}}


# {{{ test_compute_assortativity_arcagni


def test_compute_assortativity_arcagni_validation() -> None:
    """Input validation for compute_assortativity_arcagni."""
    from orbitkit.metrics import compute_assortativity_arcagni

    mat = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    with pytest.raises(ValueError, match="not square"):
        compute_assortativity_arcagni(np.ones((3, 4)))

    with pytest.raises(ValueError, match="not defined for empty matrices"):
        compute_assortativity_arcagni(np.zeros((0, 0)))

    with pytest.raises(ValueError, match="'eps' must be positive"):
        compute_assortativity_arcagni(mat, eps=-1.0)

    with pytest.raises(ValueError, match="zero diagonal"):
        compute_assortativity_arcagni(np.array([[1.0, 1.0], [1.0, 0.0]]))

    with pytest.raises(ValueError, match="negative entries"):
        compute_assortativity_arcagni(
            np.array([
                [0.0, 1.0, -1.0],
                [1.0, 0.0, 1.0],
                [-1.0, 1.0, 0.0],
            ])
        )

    with pytest.raises(ValueError, match="not symmetric"):
        compute_assortativity_arcagni(
            np.array([
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ])
        )


def test_compute_assortativity_arcagni_known_value() -> None:
    """Hand-computed coefficient on a 4-node positive graph."""
    from orbitkit.metrics import compute_assortativity_arcagni

    mat = np.array([
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    assert compute_assortativity_arcagni(mat) == pytest.approx(-5.0 / 7.0)


def test_compute_assortativity_arcagni_star_is_disassortative() -> None:
    """A star graph is perfectly disassortative (rho = -1)."""
    from orbitkit.metrics import compute_assortativity_arcagni

    mat = np.array([
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    assert compute_assortativity_arcagni(mat) == pytest.approx(-1.0)


def test_compute_assortativity_arcagni_undefined_nan() -> None:
    """Assortativity is NaN for zero and uniform-strength graphs."""
    from orbitkit.metrics import compute_assortativity_arcagni

    # all-zero matrix: omega == 0
    assert np.isnan(compute_assortativity_arcagni(np.zeros((3, 3))))

    # complete graph with uniform weights: all strengths equal
    mat = np.ones((4, 4)) - np.eye(4)
    assert np.isnan(compute_assortativity_arcagni(mat))


def test_compute_assortativity_arcagni_integer_dtype() -> None:
    """Integer matrices exercise the 'eps' fallback when finfo fails."""
    from orbitkit.metrics import compute_assortativity_arcagni

    mat = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ])
    assert compute_assortativity_arcagni(mat) == pytest.approx(-5.0 / 7.0)


def test_compute_assortativity_arcagni_explicit_eps_matches_default() -> None:
    """Passing eps explicitly does not change the result."""
    from orbitkit.metrics import compute_assortativity_arcagni

    mat = np.array([
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    r_default = compute_assortativity_arcagni(mat)
    r_explicit = compute_assortativity_arcagni(mat, eps=1.0e-8)
    assert r_default == pytest.approx(r_explicit)


def test_compute_assortativity_arcagni_scale_invariance() -> None:
    """Scaling all weights by a positive constant leaves rho unchanged."""
    from orbitkit.metrics import compute_assortativity_arcagni

    mat = np.array([
        [0.0, 2.0, 1.0, 0.5],
        [2.0, 0.0, 3.0, 1.0],
        [1.0, 3.0, 0.0, 2.0],
        [0.5, 1.0, 2.0, 0.0],
    ])
    r = compute_assortativity_arcagni(mat)
    r_scaled = compute_assortativity_arcagni(5.0 * mat)
    assert r == pytest.approx(r_scaled)


def test_compute_assortativity_arcagni_permutation_invariance() -> None:
    """Relabeling nodes must not change the coefficient."""
    from orbitkit.metrics import compute_assortativity_arcagni

    rng = np.random.default_rng(42)
    mat = np.array([
        [0.0, 2.0, 1.0, 0.5],
        [2.0, 0.0, 3.0, 1.0],
        [1.0, 3.0, 0.0, 2.0],
        [0.5, 1.0, 2.0, 0.0],
    ])
    perm = rng.permutation(mat.shape[0])
    matp = mat[perm][:, perm]

    r = compute_assortativity_arcagni(mat)
    rp = compute_assortativity_arcagni(matp)
    assert r == pytest.approx(rp)


def test_compute_assortativity_arcagni_in_range() -> None:
    """Coefficient stays in [-1, 1] (or NaN) on random weighted graphs."""
    from orbitkit.metrics import compute_assortativity_arcagni

    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(4, 12))
        mat = rng.uniform(0.0, 1.0, size=(n, n))
        mat = (mat + mat.T) / 2.0
        np.fill_diagonal(mat, 0.0)

        r = compute_assortativity_arcagni(mat)
        assert np.isnan(r) or -1.0 <= r <= 1.0


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
