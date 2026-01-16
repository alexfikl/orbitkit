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


# {{{ test_codegen_jax


@pytest.mark.parametrize(
    ("module_name", "model_name"),
    [
        ("fitzhugh_nagumo", "Omelchenko2019Figure4a"),
        ("kuramoto", "Abrams2008Figure2a"),
        ("pfeuty", "Pfeuty2007Figure2cl"),
        ("wang_buzsaki", "WangBuzsaki1996Figure3a"),
        ("wang_rinzel", "WangRinzel1992Figure1a"),
        ("wang_rinzel", "WangRinzel1992Figure4a"),
    ],
)
def test_codegen_numpy(module_name: str, model_name: str) -> None:
    """Check that the generated code works for these models."""

    pytest.importorskip("pymbolic")
    pytest.importorskip("jax")

    from testlib import get_model_from_module

    n = 32
    rng = np.random.default_rng(seed=42)

    model = get_model_from_module(module_name, model_name, n)
    d = len(model.variables)

    from orbitkit.codegen.jax import JaxTarget

    target = JaxTarget()
    source = target.lambdify_model(model, n)
    assert source is not None

    ys = rng.random(d * n)
    result = source(0.0, ys)

    assert np.all(np.isfinite(result))
    assert result.shape == (d * n,)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
