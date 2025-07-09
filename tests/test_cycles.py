# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from orbitkit.utils import module_logger

log = module_logger(__name__)


def test_limit_cycle_psd() -> None:
    rng = np.random.default_rng(seed=32)

    # generate some random data with some noise
    a = 0.0
    b = 0.01
    theta = np.linspace(0.0, 32 * np.pi, 1024)
    x = np.stack([
        np.sin(theta) * np.exp(-a * theta) + b * rng.normal(size=theta.shape),
        np.cos(theta) * np.exp(-a * theta) + b * rng.normal(size=theta.shape),
        ])

    import orbitkit.cycles as okc

    result = okc.characterize_limit_cycle_psd(x, eps=2.0e-3)
    assert result.is_limit_cycle


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
