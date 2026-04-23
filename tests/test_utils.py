# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
import time

import numpy as np
import pytest

from orbitkit.utils import module_logger
from orbitkit.visualization import set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)
set_plotting_defaults()

# {{{ test_estimate_scaling


@pytest.mark.parametrize(
    ("p", "q"),
    [
        (1.0, 0.0),
        (1.0, 1.0),
        (1.5, 0.0),
        (1.0, 2.0),
    ],
)
def test_estimate_scaling(p: float, q: float) -> None:
    from orbitkit.utils import estimate_scaling, solve_scaling_line

    rng = np.random.default_rng(seed=42)
    sigma = 0.1
    a0, a1 = 1.5, 5.0
    _, b1 = 0.0, 15.0

    for _ in range(16):
        # NOTE: generating a in [0, 5] and then b in [a + 1, 15]
        xa = a0 + (a1 - a0) * rng.random()
        xb = 2.0 + xa + (b1 - xa - 2.0) * rng.random()

        # construct y of the expected shape and add some noise
        c = xb - xa
        x = np.linspace(xa, xb, 32)
        y = c * x**p * np.log(x) ** q
        y += sigma * rng.random(y.shape)

        c_est, p_est, q_est = estimate_scaling(x, y)
        log.info(
            "(c, p, q): [%g, %g, %g] estimate [%g, %g, %g]",
            c,
            p,
            q,
            c_est,
            p_est,
            q_est,
        )

        assert abs(c - c_est) < 2.0
        assert abs(p - p_est) < 1.0
        assert abs(q - q_est) < 1.0

        xmin, xmax = x[0], x[-1]
        ymin, ymax = y[0], y[-1]

        (xmin_est, _), _ = solve_scaling_line(xmax, ymin, ymax, order=(p, q))
        log.info("xmin:      %g estimate %g", xmin, xmin_est)
        assert abs(xmin - xmin_est) < sigma


# }}}


# {{{ test_tic_toc_timer


def test_tic_toc_timer() -> None:
    from orbitkit.utils import TicTocTimer

    n = 5
    tt = TicTocTimer()
    for _ in range(n):
        tt.tic()
        time.sleep(1)
        tt.toc()

        assert tt.t_wall >= 1.0

    assert tt.n_calls == n
    assert tt.t_avg >= 1.0
    assert tt.t_sqr < 0.01

    log.info("timer: %s", tt)
    log.info("short: %s", tt.short())
    log.info("stats: %s", tt.stats())


# }}}


# {{{ test_block_timer


def test_block_timer() -> None:
    from orbitkit.utils import BlockTimer

    with BlockTimer("testing") as bt:
        time.sleep(1)

    assert bt.t_wall >= 1.0
    log.info("timer: %s", bt)


# }}}


# {{{ find_common_path


def test_find_common_path() -> None:
    from orbitkit.utils import find_common_path

    # no arguments
    assert find_common_path() is None

    # single path returned as-is
    p = pathlib.Path("path/to/foo_bar_baz.npy")
    assert find_common_path(p) == p

    # basic common prefix
    assert find_common_path(
        pathlib.Path("alphaX1"),
        pathlib.Path("alphaY2"),
        pathlib.Path("alphaZ3"),
    ) == pathlib.Path("alpha")

    # common substrings joined with the default separator
    assert find_common_path(
        pathlib.Path("foo_x_bar_123_baz"),
        pathlib.Path("foo_y_bar_456_baz"),
        pathlib.Path("foo_z_bar_789_baz"),
    ) == pathlib.Path("foo_bar_baz")

    # custom separator
    assert find_common_path(
        pathlib.Path("pre_x_mid_y_post.out"),
        pathlib.Path("pre_a_mid_b_post.out"),
        sep="-",
    ) == pathlib.Path("pre_-_mid_-_post.out")

    # parent and suffix are preserved
    assert find_common_path(
        pathlib.Path("/data/run01_alpha.csv"),
        pathlib.Path("/data/run02_beta.csv"),
        pathlib.Path("/data/run03_gamma.csv"),
    ) == pathlib.Path("/data/run0_a.csv")

    # different parents -> None
    assert (
        find_common_path(
            pathlib.Path("/a/foo_bar.txt"),
            pathlib.Path("/b/foo_baz.txt"),
        )
        is None
    )

    # different extensions -> None
    assert (
        find_common_path(
            pathlib.Path("foo_bar.txt"),
            pathlib.Path("foo_baz.csv"),
        )
        is None
    )

    # identical paths
    p = pathlib.Path("result_001.npy")
    assert find_common_path(p, p, p) == p

    # no common substring raises ValueError
    with pytest.raises(ValueError, match="empty name"):
        find_common_path(pathlib.Path("aaa.dat"), pathlib.Path("bbb.dat"))


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
