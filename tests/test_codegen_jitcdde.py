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
    from orbitkit.codegen.jitcdde import JiTCDDECompiledCode

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)
set_plotting_defaults()


# {{{ test_codegen_jitcdde


def _make_dde_from_name(
    module_name: str,
    model_name: str,
    max_delay: float,
    *,
    module_location: pathlib.Path | None = None,
) -> JiTCDDECompiledCode:
    from testlib import get_model_from_module

    n = 1 if module_name == "hiv" else 2
    model = get_model_from_module(module_name, model_name, n, delayed=True)

    from orbitkit.models import transform_distributed_delay_model

    ext_model = transform_distributed_delay_model(model, n)
    d = len(ext_model.variables)

    from orbitkit.codegen.jitcdde import JiTCDDETarget

    target = JiTCDDETarget()
    code = target.generate_model_code(ext_model, n)

    from orbitkit.utils import tictoc

    with tictoc(f"{module_name}[{model_name}]"):
        integrator = target.compile(
            code,
            max_delay=max_delay,
            module_location=module_location,
            debug=True,
        )

    log.info("\n%s", integrator.f)
    assert integrator.f.shape == (d * n,)

    integrator.set_initial_conditions(np.ones(d * n), 0.0)
    integrator.adjust_diff()

    return integrator


@pytest.mark.parametrize(
    ("module_name", "model_name", "max_delay"),
    [
        ("hiv", "CulshawRuanWebb2003Figure44", 1.0),
        ("wilson_cowan", "CustomSet1", 0.5),
    ],
)
def test_codegen_jitcdde(module_name: str, model_name: str, max_delay: float) -> None:
    """Check that the generated code works for these models."""

    pytest.importorskip("pymbolic")
    pytest.importorskip("jitcdde")

    dde = _make_dde_from_name(module_name, model_name, max_delay)
    for t in [-max_delay, 0.0, 0.01, 0.02]:
        dde.integrate(max_delay + t)


# }}}

# {{{ test_codegen_jitcdde_cache


def test_codegen_jitcdde_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that the generated code works for these models."""

    pytest.importorskip("pymbolic")
    pytest.importorskip("jitcdde")

    max_delay = 1.0
    module_location = (
        pathlib.Path(tempfile.gettempdir()) / "jitcdde_orbitkit_codegen.so"
    )

    dde = _make_dde_from_name("hiv", "CulshawRuanWebb2003Figure44", max_delay)
    dde = _make_dde_from_name("hiv", "CulshawRuanWebb2003Figure44", max_delay)
    assert not module_location.exists()
    dde = _make_dde_from_name(
        "hiv",
        "CulshawRuanWebb2003Figure44",
        max_delay,
        module_location=module_location,
    )
    assert module_location.exists()
    dde = _make_dde_from_name(
        "hiv",
        "CulshawRuanWebb2003Figure44",
        max_delay,
        module_location=module_location,
    )
    assert module_location.exists()

    def dummy_compile_c(self) -> None:
        raise AssertionError()

    # FIXME: not sure this is sufficient to check no more compilation is
    # happening?
    monkeypatch.setattr(dde.dde, "_compile_C", dummy_compile_c)
    monkeypatch.setattr(dde.dde, "compile_C", dummy_compile_c)

    for t in [-max_delay, 0.0, 0.01, 0.02]:
        dde.integrate(max_delay + t)

    if module_location.exists():
        module_location.unlink()


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
