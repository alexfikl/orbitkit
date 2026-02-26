# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from orbitkit.utils import enable_test_plotting, module_logger
from orbitkit.visualization import figure, set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)
set_plotting_defaults()


# {{{ test_cycles_harmonic


@pytest.mark.parametrize("b", [0.1, 0.01, 0.001])
def test_cycles_harmonic(b: float) -> None:
    rng = np.random.default_rng(seed=32)

    # generate some random data with some noise
    a = -0.001
    theta = np.linspace(0.0, 32 * np.pi, 2048)
    x = np.sin(theta) * np.exp(-a * theta) + b * rng.normal(size=theta.shape)

    from orbitkit.cycles import detect_cycle_harmonic

    result = detect_cycle_harmonic(x, nwindows=6)

    error = 1.0 - result.harmonic_energy / result.total_energy
    log.info("Error: %.8e (%.8e)", error, b)
    assert error < 15.0 * b

    if not enable_test_plotting():
        return

    with figure(TEST_DIRECTORY / "test_cycles_harmonic_psd", normalize=True) as fig:
        ax = fig.gca()

        ax.plot(result.freq, result.psd)
        ax.set_xlabel("$k$")
        ax.set_ylabel("PSD")


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
