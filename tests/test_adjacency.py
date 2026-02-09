# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from orbitkit.utils import module_logger
from orbitkit.visualization import set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)
set_plotting_defaults()


# {{{ test_generate_adjacency_feed_forward


def test_generate_adjacency_feed_forward() -> None:
    from orbitkit.adjacency import generate_adjacency_feed_forward

    n = 100
    dtype = np.dtype(np.uint8)

    mat = generate_adjacency_feed_forward(n, dtype=dtype)
    assert mat.shape == (n, n)
    assert mat.dtype == dtype
    assert np.all(np.diag(mat) == 0)

    assert np.all(np.triu(mat) == 0)

    mat = generate_adjacency_feed_forward(1, dtype=dtype)
    assert mat.shape == (1, 1)
    assert np.all(mat == 0)


# }}}


# {{{ test_generate_adjacency_ring


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_generate_adjacency_ring(k: int) -> None:
    from orbitkit.adjacency import generate_adjacency_ring

    n = 100
    dtype = np.dtype(np.uint8)

    mat = generate_adjacency_ring(n, k=k, dtype=dtype)
    assert mat.shape == (n, n)
    assert mat.dtype == dtype
    assert np.all(np.diag(mat) == 0)

    # check symmetry
    assert np.array_equal(mat, mat.T)

    # check degree
    degree = np.sum(mat, axis=1)
    assert np.all(degree == (0 if k == 0 else (2 * k)))

    # check circulant
    for i in range(n):
        assert np.array_equal(mat[i], np.roll(mat[0], i))

    # check that it ignores k
    mat = generate_adjacency_ring(1, k=k, dtype=dtype)
    assert mat.shape == (1, 1)


def test_generate_adjacency_ring_edge_cases() -> None:
    from orbitkit.adjacency import compute_graph_triangles, generate_adjacency_ring

    n = 100
    dtype = np.dtype(np.uint8)

    with pytest.raises(ValueError, match="invalid"):
        generate_adjacency_ring(n, k=-1, dtype=dtype)

    with pytest.raises(ValueError, match="invalid"):
        generate_adjacency_ring(n, k=n, dtype=dtype)

    with pytest.raises(ValueError, match="negative"):
        generate_adjacency_ring(-n, dtype=dtype)

    mat = generate_adjacency_ring(n, k=1)
    assert compute_graph_triangles(mat) == 0


# }}}


# {{{ test_generate_adjacency_bus


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_generate_adjacency_bus(k: int) -> None:
    from orbitkit.adjacency import generate_adjacency_bus

    n = 100
    dtype = np.dtype(np.uint8)

    mat = generate_adjacency_bus(n, k=k, dtype=dtype)
    assert mat.shape == (n, n)
    assert mat.dtype == dtype
    assert np.all(np.diag(mat) == 0)

    # check symmetry
    assert np.array_equal(mat, mat.T)

    # check degree
    # FIXME: this could be made more exact, we know the actual degrees
    degree = np.sum(mat, axis=1)
    assert np.all(degree <= (0 if k == 0 else (2 * k)))

    # check that it ignores k
    mat = generate_adjacency_bus(1, k=k, dtype=dtype)
    assert mat.shape == (1, 1)


def test_generate_adjacency_bus_edge_cases() -> None:
    from orbitkit.adjacency import compute_graph_triangles, generate_adjacency_bus

    n = 100
    dtype = np.dtype(np.uint8)

    with pytest.raises(ValueError, match="invalid"):
        generate_adjacency_bus(n, k=-1, dtype=dtype)

    with pytest.raises(ValueError, match="invalid"):
        generate_adjacency_bus(n, k=n, dtype=dtype)

    with pytest.raises(ValueError, match="negative"):
        generate_adjacency_bus(-n, dtype=dtype)

    mat = generate_adjacency_bus(n, k=1)
    assert compute_graph_triangles(mat) == 0


# }}}


# {{{ test_generate_adjacency_star


def test_generate_adjacency_star() -> None:
    from orbitkit.adjacency import generate_adjacency_star

    n = 100
    dtype = np.dtype(np.uint8)

    mat = generate_adjacency_star(n, dtype=dtype)
    assert mat.shape == (n, n)
    assert mat.dtype == dtype
    assert np.all(np.diag(mat) == 0)

    # check symmetry
    assert np.array_equal(mat, mat.T)

    # check degree
    degree = np.sum(mat, axis=1)
    assert degree[0] == n - 1
    assert np.all(degree[1:] == 1)

    # check edges
    edges = np.sum(mat) // 2
    assert edges == n - 1

    # check single node
    mat = generate_adjacency_star(1, dtype=dtype)
    assert mat.shape == (1, 1)


def test_generate_adjacency_star_tree() -> None:
    from orbitkit.adjacency import generate_adjacency_star_tree

    n = 100
    dtype = np.dtype(np.uint8)

    mat = generate_adjacency_star_tree(n, dtype=dtype)
    assert mat.shape == (n, n)
    assert mat.dtype == dtype
    assert np.all(np.diag(mat) == 0)

    # check symmetry
    assert np.array_equal(mat, mat.T)

    # check single node
    mat = generate_adjacency_star_tree(1, dtype=dtype)
    assert mat.shape == (1, 1)

    # check edge cases
    with pytest.raises(ValueError, match="negative"):
        mat = generate_adjacency_star_tree(-1, dtype=dtype)

    with pytest.raises(ValueError, match="negative"):
        mat = generate_adjacency_star_tree(n, nhubs=-4, dtype=dtype)

    with pytest.raises(ValueError, match="higher"):
        mat = generate_adjacency_star_tree(5, nhubs=7, dtype=dtype)


# }}}


# {{{ test_generate_adjacency_lattice


@pytest.mark.parametrize("n", [32, 49, 64, 95])
def test_generate_adjacency_lattice(n: int) -> None:
    from orbitkit.adjacency import (
        _find_equal_factors,  # noqa: PLC2701
        generate_adjacency_lattice,
    )

    m, p = _find_equal_factors(n)
    dtype = np.dtype(np.uint8)

    mat = generate_adjacency_lattice(n, dtype=dtype)
    assert mat.shape == (n, n)
    assert mat.dtype == dtype
    assert np.all(np.diag(mat) == 0)

    # check symmetry
    assert np.array_equal(mat, mat.T)

    # check edge count
    assert np.sum(mat) // 2 == (m * (p - 1) + (m - 1) * p)


@pytest.mark.parametrize(("n", "m"), [(15, 3), (15, 1), (1, 15), (7, 7)])
def test_generate_adjacency_lattice_both(n: int, m: int) -> None:
    from orbitkit.adjacency import generate_adjacency_lattice

    dtype = np.dtype(np.uint8)

    mat = generate_adjacency_lattice(n, m, dtype=dtype)
    assert mat.shape == (n * m, n * m)
    assert mat.dtype == dtype
    assert np.all(np.diag(mat) == 0)

    # check symmetry
    assert np.array_equal(mat, mat.T)

    # check edge count
    assert np.sum(mat) // 2 == (n * (m - 1) + (n - 1) * m)


# }}}


# {{{ test_generate_adjacency_erdos_renyi


@pytest.mark.parametrize(("n", "p"), [(100, 0.25), (500, 0.05), (300, 0.95)])
@pytest.mark.parametrize("symmetric", [True, False])
def test_generate_erdos_renyi_probability(n: int, p: float, symmetric: bool) -> None:  # noqa: FBT001
    """Check that the generated matrix has the desired probability distribution."""
    from orbitkit.adjacency import generate_adjacency_erdos_renyi

    dtype = np.dtype(np.uint8)
    rng = np.random.default_rng(seed=42)

    for _ in range(32):
        A = generate_adjacency_erdos_renyi(
            n, p=p, symmetric=symmetric, dtype=dtype, rng=rng
        )
        assert A.shape == (n, n)
        assert A.dtype == dtype
        assert np.all(np.diag(A) == 0)

        M = n * (n - 1) // 2
        E = np.sum(np.tril(A, k=-1))
        phat = E / M

        sigma = np.sqrt(p * (1 - p) / M)
        zscore = abs(phat - p) / sigma

        log.info("p %.2f phat %.5f zscore %.8e", p, phat, zscore)
        assert zscore < 4.0


@pytest.mark.parametrize(("n", "k"), [(100, 25), (500, 10), (300, 200)])
@pytest.mark.parametrize("symmetric", [True, False])
def test_generate_erdos_renyi_degree(n: int, k: int, symmetric: bool) -> None:  # noqa: FBT001
    """Check that the generated matrix has the desired node degree."""
    from orbitkit.adjacency import generate_adjacency_erdos_renyi

    dtype = np.dtype(np.uint8)
    rng = np.random.default_rng(seed=42)

    degree = np.zeros(32)
    for i in range(degree.size):
        A = generate_adjacency_erdos_renyi(
            n, k=k, symmetric=symmetric, dtype=dtype, rng=rng
        )
        assert A.shape == (n, n)
        assert A.dtype == dtype
        assert np.all(np.diag(A) == 0)

        E = np.sum(A, axis=1)
        degree[i] = np.mean(E)
        log.info("K %d Khat %.2f", k, np.mean(E))

    mu = np.mean(degree)
    sigma = np.std(degree, ddof=1)
    zscore = (k - mu) / sigma

    log.info("K %d Khat %.2f zscore %.8e", k, mu, zscore)
    assert abs(zscore) < 4.0


def test_generate_adjacency_erdos_renyi_edge_cases() -> None:
    from orbitkit.adjacency import generate_adjacency_erdos_renyi

    n = 100
    rng = np.random.default_rng(seed=42)

    with pytest.raises(ValueError, match="negative"):
        _ = generate_adjacency_erdos_renyi(-1, rng=rng)

    with pytest.raises(ValueError, match="both"):
        _ = generate_adjacency_erdos_renyi(n, p=0.5, k=10, rng=rng)

    with pytest.raises(ValueError, match="not in"):
        _ = generate_adjacency_erdos_renyi(n, p=1.5, rng=rng)

    with pytest.raises(ValueError, match="more edges"):
        _ = generate_adjacency_erdos_renyi(n, k=101, rng=rng)

    with pytest.raises(ValueError, match="more edges"):
        _ = generate_adjacency_erdos_renyi(n, k=-12, rng=rng)

    # check p = 0
    mat = generate_adjacency_erdos_renyi(n, p=0, symmetric=True, rng=rng)
    assert mat.shape == (n, n)
    assert np.all(mat == 0)

    mat = generate_adjacency_erdos_renyi(n, p=0, symmetric=False, rng=rng)
    assert mat.shape == (n, n)
    assert np.all(mat == 0)

    # check p = 1
    mat = generate_adjacency_erdos_renyi(n, p=1.0, symmetric=True, rng=rng)
    assert np.all(np.sum(mat, axis=1) == n - 1)

    mat = generate_adjacency_erdos_renyi(n, p=1.0, symmetric=False, rng=rng)
    assert np.all(np.sum(mat, axis=1) == n - 1)


# }}}


# {{{ test_generate_adjacency_strogatz_watts


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_generate_adjacency_strogatz_watts(k: int) -> None:
    from orbitkit.adjacency import generate_adjacency_strogatz_watts

    n = 100
    dtype = np.dtype(np.uint8)
    rng = np.random.default_rng(seed=42)

    for p in (0, 0.1, 0.5, 0.9, 1.0):
        mat = generate_adjacency_strogatz_watts(n, k=k, p=p, dtype=dtype, rng=rng)
        assert mat.shape == (n, n)
        assert mat.dtype == dtype
        assert np.all(np.diag(mat) == 0)

        # check symmetry
        assert np.array_equal(mat, mat.T)

        # check degree
        degree = np.sum(mat, axis=1)
        if p == 0:
            # NOTE: this is just a ring network for p = 0
            assert np.all(degree == (0 if k == 0 else (2 * k)))
        else:
            # NOTE: we cannot know the exact degree, e.g. all edges may move
            # to the same node so that we have degree = n - 1.
            assert np.all(degree <= 4 * 2 * k)


# }}}


# {{{ test_generate_adjacency_gap_junction


@pytest.mark.parametrize(
    ("n", "m"),
    [
        (100, 2),
        (100, 3),
        (100, 4),
        (100, 5),
        (200, 6),
        (300, 7),
    ],
)
def test_generate_gap_junction_probability(n: int, m: int) -> None:
    """Check that the cluster sizes have the desired statistics."""
    from orbitkit.adjacency import (
        _generate_random_gap_junction_clusters,  # noqa: PLC2701
    )

    rng = np.random.default_rng(seed=42)

    avgsize = 9
    maxsize = 21
    maxiter = 512

    cavg = np.empty(maxiter)
    cmax = np.empty(maxiter)

    for i in range(maxiter):
        clusters = _generate_random_gap_junction_clusters(
            rng,
            n,
            m,
            alpha=1.0,
            avgsize=avgsize,
            maxsize=maxsize,
            maxiter=maxiter,
        )

        assert np.sum(clusters) <= n, np.sum(clusters)
        cavg[i] = np.mean(clusters)
        cmax[i] = np.max(clusters)

    mu = np.mean(cavg)

    log.info(
        "mean %d mean est %.2f max %d max est %.f", avgsize, mu, maxsize, np.max(cmax)
    )

    assert abs(avgsize - mu) < 0.5
    assert np.max(cmax) <= maxsize


@pytest.mark.parametrize(
    ("n", "m"),
    [
        (100, 2),
        (100, 3),
        (100, 4),
        (100, 5),
        (200, 6),
        (300, 7),
    ],
)
def test_generate_adjacency_gap_junctions(n: int, m: int) -> None:
    from orbitkit.adjacency import generate_adjacency_gap_junctions

    dtype = np.dtype(np.uint8)
    rng = np.random.default_rng(seed=42)

    mat = generate_adjacency_gap_junctions(n, m, dtype=dtype, rng=rng)
    assert mat.shape == (n, n)
    assert mat.dtype == dtype
    assert np.all(np.diag(mat) == 0)

    # check symmetry
    assert np.array_equal(mat, mat.T)


# }}}


# {{{ test_generate_adjacency_fractal


@pytest.mark.parametrize(
    ("base", "nlevels"),
    [
        ("101", 1),
        ("101", 3),
        ("110011", 2),
        ("101110", 3),
    ],
)
def test_expand_pattern(base: str, nlevels: int) -> None:
    from orbitkit.adjacency import _expand_pattern  # noqa: PLC2701

    dtype = np.dtype(np.uint8)
    pattern = _expand_pattern(base, nlevels, dtype=dtype)

    assert pattern.shape == (len(base) ** nlevels,)
    assert pattern.sum() == base.count("1") ** nlevels


@pytest.mark.parametrize("nlevels", [0, 1, 4])
def test_generate_adjacency_fractal(nlevels: int) -> None:
    from orbitkit.adjacency import generate_adjacency_fractal

    dtype = np.dtype(np.uint8)
    n = 3**nlevels

    mat = generate_adjacency_fractal("000", nlevels=nlevels, dtype=dtype)
    assert mat.shape == (n, n)
    assert mat.dtype == dtype
    assert np.all(np.diag(mat) == 0)
    assert np.sum(mat) == 0

    mat = generate_adjacency_fractal("111", nlevels=nlevels, dtype=dtype)
    assert np.all(np.diag(mat) == 0)
    assert np.all(np.sum(mat, axis=1) == n - 1)

    mat = generate_adjacency_fractal("101", nlevels=nlevels, dtype=dtype)
    assert np.all(np.diag(mat) == 0)

    # if not ENABLE_VISUAL:
    #     return

    # if n <= 1:
    #     return

    # import networkx as nx
    # import nxviz as nv

    # # NOTE: only plot the first row
    # mat[0, 0] = 1
    # mat[1:, :] = 0
    # G = nx.from_numpy_array(mat)

    # for i, b in enumerate(mat[0, :]):
    #     G.nodes[i]["group"] = f"{b}"

    # from orbitkit.visualization import figure

    # with figure(
    #     TEST_DIRECTORY / f"test_generate_adjacency_fractal_{nlevels}",
    #     normalize=True,
    # ) as fig:
    #     ax = fig.gca()

    #     import matplotlib.pyplot as mp

    #     colors = mp.rcParams["axes.prop_cycle"].by_key()["color"]
    #     palette = {f"{b}": colors[b] for b in (0, 1)}

    #     nv.circos(
    #         G,
    #         node_color_by="group",
    #         node_palette=palette,
    #     )

    #     pattern = "".join(str(b) for b in mat[0, :])
    #     ax.set_title(pattern)


# }}}


# {{{ test_make_graph_laplacian_undirected

ADJACENCY_SYMMETRIC = frozenset({
    "lattice",
    "strogatzwatts",
    "ring1",
    "ring",
    "erdosrenyi",
    "bus2",
    "configuration",
    "ring2",
    "bus",
    "startree",
    "bus1",
})


@pytest.mark.parametrize("normalize", [True, False])
def test_make_graph_laplacian_undirected(normalize: bool) -> None:  # noqa: FBT001
    from orbitkit.adjacency import (
        make_adjacency_matrix_from_name,
        make_graph_laplacian_undirected,
    )

    rng = np.random.default_rng(seed=42)
    atol = 2.0e-14

    n = 128
    x = np.ones(n)

    for topology in ADJACENCY_SYMMETRIC:
        mat = make_adjacency_matrix_from_name(n, topology, rng=rng)
        L = make_graph_laplacian_undirected(mat, normalize=normalize)

        if normalize:
            D = np.diag(np.sqrt(np.sum(mat, axis=1)))
            error = np.linalg.norm(L @ D @ x)
        else:
            error = np.linalg.norm(L @ x)
        assert error < atol

        assert np.allclose(L, L.T)

        eigs = np.linalg.eigvals(L)
        assert np.all(eigs > -atol)

        if normalize:
            assert np.all(eigs < 2.0 + atol)


# }}}


# {{{ test_make_graph_laplacian_directed


@pytest.mark.parametrize("out", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_make_graph_laplacian_directed(out: bool, normalize: bool) -> None:  # noqa: FBT001
    from orbitkit.adjacency import (
        ADJACENCY_TYPES,
        make_adjacency_matrix_from_name,
        make_graph_laplacian_directed,
    )

    rng = np.random.default_rng(seed=42)
    atol = 4.0e-14
    n = 128

    for topology in ADJACENCY_TYPES:
        mat = make_adjacency_matrix_from_name(n, topology, rng=rng)
        L = make_graph_laplacian_directed(mat, out=out, normalize=normalize)
        x = np.ones(L.shape[1])

        if normalize:
            if out:  # noqa: SIM108
                error = np.linalg.norm(L @ x)
            else:
                # NOTE: not sure what the right eigenvector would be?
                error = np.linalg.norm(x @ L)
        else:  # noqa: PLR5501
            if out:  # noqa: SIM108
                error = np.linalg.norm(L @ x)
            else:
                error = np.linalg.norm(x @ L)

        assert error < atol

        eigs = np.linalg.eigvals(L)
        assert np.all(eigs.real > -atol)


# }}}


# {{{ test_generate_graph_laplacian_weights


def test_generate_graph_laplacian_weights() -> None:
    from orbitkit.adjacency import (
        apply_graph_laplacian,
        make_adjacency_matrix_from_name,
    )
    from orbitkit.typing import Array

    def f_sin(x: Array) -> Array:
        return 2.0 + np.sin(x)

    def f_inv(x: Array) -> Array:
        return 1.0 / (2.0 + x)

    rng = np.random.default_rng(seed=42)
    atol = 4.0e-14
    n = 128

    for topology in ADJACENCY_SYMMETRIC:
        mat = make_adjacency_matrix_from_name(n, topology, rng=rng)
        W = apply_graph_laplacian(mat, f_inv)
        x = np.ones(W.shape[1])

        assert np.allclose(W, W.T)

        w_eigs = np.linalg.eigvals(W)
        assert np.all(w_eigs.real > -atol)

        error = np.linalg.norm(W @ x - f_inv(0) * x)  # ty: ignore[invalid-argument-type]
        assert error < atol


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
