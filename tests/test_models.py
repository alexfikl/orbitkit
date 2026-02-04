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

# {{{ test_wilson_cowan_fixed_points


def test_wilson_cowan_fixed_points() -> None:
    from orbitkit.models.rate_functions import SigmoidRate
    from orbitkit.models.wilson_cowan import get_wilson_cowan_fixed_points

    def sigmoid(s: SigmoidRate, x: float) -> float:
        return s.a / (1.0 + np.exp(-(x - s.theta) / s.sigma))  # ty: ignore[no-matching-overload]

    rng = np.random.default_rng(seed=42)
    rtol = 1.0e-8

    for _ in range(8):
        sigma = rng.uniform(0.0, 5.0)
        sE = SigmoidRate(1.0, 0.0, sigma)

        sigma = rng.uniform(0.0, 5.0)
        sI = SigmoidRate(1.0, 0.0, sigma)

        a, b, c, d = rng.uniform(0, 10.0, size=4)
        p, q = 0.0, 0.0

        result = get_wilson_cowan_fixed_points(sE, sI, (a, b, c, d), (p, q), rtol=rtol)

        log.info("Roots: %d", result.shape[0])
        assert 1 <= result.shape[0] <= 3

        for i in range(result.shape[0]):
            Estar, Istar = result[i]
            rE = abs(Estar - sigmoid(sE, a * Estar - b * Istar + p)) / abs(Estar)
            rI = abs(Istar - sigmoid(sI, c * Estar - d * Istar + q)) / abs(Istar)
            log.info("point (%.8e, %.8e) residuals E %.8e I %.8e", Estar, Istar, rE, rI)

            assert 0.0 <= Estar <= 1.0
            assert 0.0 <= Istar <= 1.0

            assert rE < 10 * rtol
            assert rI < 10 * rtol

    # {{{ known: 1 solution

    sE = SigmoidRate(1.0, 0.0, 2.0)
    sI = SigmoidRate(1.0, 0.0, 2.0)
    a, b, c, d = 6.0, -2.0, 6.0, -2.0
    p = q = -2.0

    result = get_wilson_cowan_fixed_points(sE, sI, (a, b, c, d), (p, q), rtol=rtol)
    log.info("Result: %s", result)

    # NOTE: we have a symmetric solution here Estar = Istar
    assert result.shape[0] == 1, result
    assert abs(result[0, 0] - result[0, 1]) < rtol

    # NOTE: "exact" solution obtained from Mathematica
    assert abs(result[0, 0] - 0.9406126803877602) < rtol

    # }}}

    # {{{ known: 2 solutions

    sE = SigmoidRate(1.0, 0.0, 1.0)
    sI = SigmoidRate(1.0, 0.0, 1.0)
    a, b, c, d = 5.0, -2.0, 5.0, -2.0
    # FIXME: this was mostly obtained by a manual bisection method :( Need something
    # more serious to find the two-solution case
    p = q = -2.7755113894944901

    result = get_wilson_cowan_fixed_points(
        sE, sI, (a, b, c, d), (p, q), rtol=rtol, method="brentq"
    )
    log.info("Result: %s", result)

    # NOTE: we have a symmetric solution here Estar = Istar
    assert result.shape[0] == 2, result
    assert abs(result[0, 0] - result[0, 1]) < rtol
    assert abs(result[1, 0] - result[1, 1]) < rtol

    # NOTE: "exact" solution obtained from Mathematica
    assert abs(result[0, 0] - 0.1726731621447889) < 2.0 * rtol
    assert abs(result[1, 0] - 0.9838836642147963) < rtol

    # }}}

    # {{{ known: 3 solutions

    sE = SigmoidRate(1.0, 0.0, 1.0)
    sI = SigmoidRate(1.0, 0.0, 1.0)
    a, b, c, d = 5.0, -2.0, 5.0, -2.0
    p = q = -3.0

    result = get_wilson_cowan_fixed_points(
        sE, sI, (a, b, c, d), (p, q), rtol=rtol, method="brentq"
    )
    log.info("Result: %s", result)

    # NOTE: we have a symmetric solution here Estar = Istar
    assert result.shape[0] == 3, result
    assert abs(result[0, 0] - result[0, 1]) < rtol
    assert abs(result[1, 0] - result[1, 1]) < rtol
    assert abs(result[2, 0] - result[2, 1]) < rtol

    # NOTE: "exact" solution obtained from Mathematica
    assert abs(result[0, 0] - 0.08035781815535596) < rtol
    assert abs(result[1, 0] - 0.3225798774191962) < rtol
    assert abs(result[2, 0] - 0.9792620601153998) < rtol

    # }}}


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
