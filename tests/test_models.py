# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
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

# {{{ test_wilson_cowan_fixed_point


def test_wilson_cowan_fixed_point() -> None:
    from orbitkit.models.rate_functions import SigmoidRate
    from orbitkit.models.wilson_cowan import get_wilson_cowan_fixed_point

    def sigmoid(s: SigmoidRate, x: float) -> float:
        return s.a / (1.0 + np.exp(-(x - s.theta) / s.sigma))  # ty: ignore[no-matching-overload]

    rng = np.random.default_rng(seed=42)
    rtol = 1.0e-8

    for _ in range(32):
        theta = rng.uniform(-5.0, 5.0)
        sigma = rng.uniform(0.0, 5.0)
        sE = SigmoidRate(1.0, theta, sigma)

        theta = rng.uniform(-5.0, 5.0)
        sigma = rng.uniform(0.0, 5.0)
        sI = SigmoidRate(1.0, theta, sigma)

        a, b, c, d = rng.uniform(0, 10.0, size=4)
        p, q = rng.uniform(-5.0, 5.0, size=2)

        Estar, Istar = get_wilson_cowan_fixed_point(
            sE, sI, (a, b, c, d), (p, q), rtol=rtol
        )

        rE = abs(Estar - sigmoid(sE, a * Estar - b * Istar + p)) / abs(Estar)
        rI = abs(Istar - sigmoid(sI, c * Estar - d * Istar + q)) / abs(Istar)
        log.info("point (%.8e, %.8e) residuals E %.8e I %.8e", Estar, Istar, rE, rI)

        assert 0.0 <= Estar <= 1.0
        assert 0.0 <= Istar <= 1.0

        assert rE < rtol
        assert rI < rtol


# }}}

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
