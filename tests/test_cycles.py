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


# {{{ test_cycles_welch_psd


@pytest.mark.parametrize("method", ["welch", "lombscargle"])
def test_cycles_welch_psd(method: str) -> None:
    rng = np.random.default_rng(seed=32)

    # generate some random data with some noise
    a = -0.001
    b = 0.001
    theta = np.linspace(0.0, 32 * np.pi, 2048)
    x = np.stack([
        np.sin(theta) * np.exp(-a * theta) + b * rng.normal(size=theta.shape),
        np.cos(theta) * np.exp(-a * theta) + b * rng.normal(size=theta.shape),
    ])

    import orbitkit.cycles as okc

    if method == "welch":
        result = okc.evaluate_welch_power_spectrum_density_deltas(x, nwindows=6)
    elif method == "lombscargle":
        result = okc.evaluate_lomb_scargle_power_spectrum_density_deltas(
            theta, x, nwindows=6
        )
    else:
        raise ValueError(f"unknown method: '{method}'")

    assert np.max(result.deltas) < 20.0 * b

    if not ENABLE_VISUAL:
        return

    from orbitkit.visualization import figure

    with figure(TEST_DIRECTORY / f"test_cycles_{method}_psd", normalize=True) as fig:
        ax = fig.gca()

        ax.plot(result.freq, result.psd)
        ax.set_xlabel("$k$")
        ax.set_ylabel("PSD")

    with figure(TEST_DIRECTORY / f"test_cycles_{method}_deltas", normalize=True) as fig:
        ax = fig.gca()

        ax.plot(result.deltas)
        ax.set_xlabel("Window")
        ax.set_ylabel(r"$\Delta$")


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
