# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import pytest

from orbitkit.utils import module_logger
from orbitkit.visualization import set_plotting_defaults

if TYPE_CHECKING:
    from orbitkit.codegen.jitcode import JiTCODECompiledCode

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)
set_plotting_defaults()


# {{{ test_codegen_jitcode


def _make_ode_from_name(
    module_name: str,
    model_name: str,
    *,
    module_location: pathlib.Path | None = None,
) -> JiTCODECompiledCode:
    from testlib import get_model_from_module

    n = 1 if module_name == "hiv" else 2

    model = get_model_from_module(module_name, model_name, n, delayed=False)
    d = len(model.variables)

    from orbitkit.codegen.jitcode import JiTCODETarget

    target = JiTCODETarget()
    code = target.generate_model_code(model, n)

    from orbitkit.utils import tictoc

    with tictoc(f"{module_name}[{model_name}]"):
        integrator = target.compile(code, module_location=module_location, debug=False)
        integrator.set_initial_conditions(np.ones(d * n), 0.0)

    log.info("\n%s", integrator.f)
    assert integrator.f.shape == (d * n,)

    return integrator


@pytest.mark.parametrize(
    ("module_name", "model_name"),
    [
        ("fitzhugh_nagumo", "Omelchenko2019Figure4a"),
        ("hiv", "CulshawRuanWebb2003Figure44"),
        ("kuramoto", "Abrams2008Figure2a"),
        ("pfeuty", "Pfeuty2007Figure2cl"),
        ("wang_buzsaki", "WangBuzsaki1996Figure3a"),
        ("wang_rinzel", "WangRinzel1992Figure1a"),
        ("wang_rinzel", "WangRinzel1992Figure4a"),
        ("wilson_cowan", "CustomSet1"),
    ],
)
def test_codegen_jitcode(module_name: str, model_name: str) -> None:
    """Check that the generated code works for these models."""

    pytest.importorskip("pymbolic")
    pytest.importorskip("jitcode")

    ode = _make_ode_from_name(module_name, model_name)
    for t in [0.0, 0.01, 0.02]:
        ode.integrate(t)


# }}}


# {{{ test_codegen_jitcode_cache


def test_codegen_jitcode_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that the generated code works for these models."""

    pytest.importorskip("pymbolic")
    pytest.importorskip("jitcode")

    module_location = (
        pathlib.Path(tempfile.gettempdir()) / "jitcode_orbitkit_codegen.so"
    )

    ode = _make_ode_from_name("hiv", "CulshawRuanWebb2003Figure44")
    ode = _make_ode_from_name("hiv", "CulshawRuanWebb2003Figure44")
    assert not module_location.exists()
    ode = _make_ode_from_name(
        "hiv",
        "CulshawRuanWebb2003Figure44",
        module_location=module_location,
    )
    assert module_location.exists()
    ode = _make_ode_from_name(
        "hiv",
        "CulshawRuanWebb2003Figure44",
        module_location=module_location,
    )
    assert module_location.exists()

    def dummy_compile_c(self) -> None:
        raise AssertionError()

    # FIXME: not sure this is sufficient to check that no mode compilation is
    # happening?
    monkeypatch.setattr(ode.ode, "_compile_C", dummy_compile_c)
    monkeypatch.setattr(ode.ode, "compile_C", dummy_compile_c)

    for t in [0.0, 0.01, 0.02]:
        ode.integrate(t)

    if module_location.exists():
        module_location.unlink()


# }}}

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
