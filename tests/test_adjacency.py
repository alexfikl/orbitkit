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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
