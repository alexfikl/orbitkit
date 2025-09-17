# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from orbitkit.utils import get_environ_boolean, module_logger, set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_boolean("ORBITKIT_ENABLE_VISUAL")

log = module_logger(__name__)
set_plotting_defaults()


# {{{ test_erdos_renyi_probability


@pytest.mark.parametrize(("n", "p"), [(100, 0.25), (500, 0.05), (300, 0.95)])
@pytest.mark.parametrize("symmetric", [True, False])
def test_erdos_renyi_probability(n: int, p: float, symmetric: bool) -> None:  # noqa: FBT001
    """Check that the generated matrix has the desired probability distribution."""
    from orbitkit.adjacency import generate_adjacency_erdos_renyi

    rng = np.random.default_rng(seed=42)
    for _ in range(32):
        A = generate_adjacency_erdos_renyi(n, p=p, symmetric=symmetric, rng=rng)

        M = n * (n + 1) // 2
        E = np.sum(np.tril(A, k=0))
        phat = E / M

        sigma = np.sqrt(p * (1 - p) / M)
        zscore = abs(phat - p) / sigma

        log.info("p %.2f phat %.5f zscore %.8e", p, phat, zscore)
        assert zscore < 4.0


@pytest.mark.parametrize(("n", "k"), [(100, 25), (500, 10), (300, 200)])
@pytest.mark.parametrize("symmetric", [True, False])
def test_erdos_renyi_degree(n: int, k: int, symmetric: bool) -> None:  # noqa: FBT001
    """Check that the generated matrix has the desired node degree."""
    from orbitkit.adjacency import generate_adjacency_erdos_renyi

    rng = np.random.default_rng(seed=42)

    degree = np.zeros(32)
    for i in range(degree.size):
        A = generate_adjacency_erdos_renyi(n, k=k, symmetric=symmetric, rng=rng)
        E = np.sum(A, axis=1)

        degree[i] = np.mean(E)
        log.info("K %d Khat %.2f", k, np.mean(E))

    mu = np.mean(degree)
    sigma = np.std(degree, ddof=1)
    zscore = (k - mu) / sigma

    log.info("K %d Khat %.2f zscore %.8e", k, mu, zscore)
    assert abs(zscore) < 4.0


# }}}


# {{{ test_gap_junction_probability


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
def test_gap_junction_probability(n: int, m: int) -> None:
    """Check that the cluster sizes have the desired statistics."""
    from orbitkit.adjacency import (
        _generate_random_gap_junction_clusters,  # noqa: PLC2701
    )

    rng = np.random.default_rng(seed=None)

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


# }}}


# {{{ test_generate_adjacency


def test_generate_adjacency_feed_forward() -> None:
    from orbitkit.adjacency import generate_adjacency_feed_forward

    n = 100
    dtype = np.dtype(np.uint8)

    mat = generate_adjacency_feed_forward(n, dtype=dtype)
    assert mat.shape == (n, n)
    assert mat.dtype == dtype
    assert np.all(np.diag(mat) == 0)

    assert np.all(np.triu(mat) == 0)


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


def test_generate_adjacency_ring_invalid() -> None:
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


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_generate_adjacency_bus(k: int) -> None:
    from orbitkit.adjacency import generate_adjacency_bus

    n = 100
    dtype = np.dtype(np.uint8)

    mat = generate_adjacency_bus(n, k=k, dtype=dtype)
    assert mat.shape == (n, n)
    assert mat.dtype == dtype

    # check symmetry
    assert np.array_equal(mat, mat.T)

    # check degree
    # FIXME: this could be made more exact, we know the actual degrees
    degree = np.sum(mat, axis=1)
    assert np.all(degree <= (0 if k == 0 else (2 * k)))


def test_generate_adjacency_bus_invalid() -> None:
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
