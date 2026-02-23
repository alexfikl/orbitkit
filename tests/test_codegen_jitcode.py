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
        # ("hiv", "CulshawRuanWebb2003Figure44"),
        ("kuramoto", "Abrams2008Figure2a"),
        ("pfeuty", "Pfeuty2007Figure2cl"),
        ("wang_buzsaki", "WangBuzsaki1996Figure3a"),
        ("wang_rinzel", "WangRinzel1992Figure1a"),
        ("wang_rinzel", "WangRinzel1992Figure4a"),
        ("wilson_cowan", "CustomSet1"),
    ],
)
def test_codegen_numpy(module_name: str, model_name: str) -> None:
    """Check that the generated code works for these models."""

    pytest.importorskip("pymbolic")
    pytest.importorskip("jitcode")

    from testlib import get_model_from_module

    n = 2

    model = get_model_from_module(module_name, model_name, n, delayed=False)
    d = len(model.variables)

    from orbitkit.codegen.jitcode import JiTCODETarget, make_input_variable

    target = JiTCODETarget()
    source_func = target.lambdify_model(model, n)
    assert source_func is not None

    import jitcode

    ys = make_input_variable(n * d)
    source = source_func(jitcode.t, ys)

    log.info("\n%s", source)
    assert source.shape == (d * n,)

    ode = target.compile(source, ys, method="RK45")
    ode.set_initial_value(np.ones(d * n), 0.0)

    for t in [0.0, 0.01, 0.02]:
        ode.integrate(t)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
