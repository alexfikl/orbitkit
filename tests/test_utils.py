# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from orbitkit.utils import get_environ_boolean, module_logger
from orbitkit.visualization import set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_boolean("ORBITKIT_ENABLE_VISUAL")

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
