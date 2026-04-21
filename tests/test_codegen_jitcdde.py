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
    *,
    max_delay: float | None = None,
    symbolic: bool = False,
    module_location: pathlib.Path | None = None,
) -> JiTCDDECompiledCode:
    from testlib import get_model_from_module

    n = 1 if module_name == "hiv" else 2
    model = get_model_from_module(
        module_name,
        model_name,
        n,
        symbolic=symbolic,
        delayed=True,
    )

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

    dde = _make_dde_from_name(module_name, model_name, max_delay=max_delay)
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

    dde = _make_dde_from_name("hiv", "CulshawRuanWebb2003Figure44", max_delay=max_delay)
    dde = _make_dde_from_name("hiv", "CulshawRuanWebb2003Figure44", max_delay=max_delay)
    assert not module_location.exists()
    dde = _make_dde_from_name(
        "hiv",
        "CulshawRuanWebb2003Figure44",
        max_delay=max_delay,
        module_location=module_location,
    )
    assert module_location.exists()
    dde = _make_dde_from_name(
        "hiv",
        "CulshawRuanWebb2003Figure44",
        max_delay=max_delay,
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


# {{{ test_codegen_jitcdde_symbolic_delay


def test_codegen_jitcdde_symbolic_delay() -> None:
    """Test JiTCDDE code generation with a symbolic delay parameter *tau*."""

    pytest.importorskip("pymbolic")
    pytest.importorskip("jitcdde")

    import orbitkit.symbolic.primitives as sym

    dde = _make_dde_from_name("hiv", "CulshawRuanWebb2003Figure44", symbolic=True)
    d = dde.f.size

    assert dde.delays == (sym.Variable("tau"),)
    assert "tau" in dde.parameters

    # 1. Integrate with tau = 0.5
    tau1 = 0.5
    dde.set_initial_conditions(np.ones(d), 0.0)
    dde.set_parameters(tau=tau1)
    dde.adjust_diff()

    assert tuple(dde.dde.delays) == (0.0, tau1)
    assert abs(dde.dde.max_delay - tau1) < 1.0e-14

    for t in [-tau1, 0.0, 0.01, 0.02]:
        dde.integrate(tau1 + t)

    # 2. Reset with a different tau and larger max_delay
    tau2 = 1.5
    dde.reset()
    dde.set_initial_conditions(np.ones(d), 0.0)
    dde.set_parameters(tau=tau2)
    dde.adjust_diff()

    assert tuple(dde.dde.delays) == (0.0, tau2)
    assert abs(dde.dde.max_delay - tau2) < 1.0e-14

    for t in [-tau2, 0.0, 0.01, 0.02]:
        dde.integrate(tau2 + t)


# }}}


# {{{ test_codegen_jitcdde_symbolic_delay_cache


def test_codegen_jitcdde_symbolic_delay_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that module_location caching works with a symbolic delay."""

    pytest.importorskip("pymbolic")
    pytest.importorskip("jitcdde")

    module_location = (
        pathlib.Path(tempfile.gettempdir()) / "jitcdde_orbitkit_symbolic_delay.so"
    )

    # First compile: writes the .so
    dde = _make_dde_from_name(
        "hiv",
        "CulshawRuanWebb2003Figure44",
        module_location=module_location,
        symbolic=True,
    )
    assert module_location.exists()

    # Reload from cache (no recompilation)
    dde = _make_dde_from_name(
        "hiv",
        "CulshawRuanWebb2003Figure44",
        module_location=module_location,
        symbolic=True,
    )
    assert module_location.exists()

    d = dde.f.size
    dde.set_initial_conditions(np.ones(d), 0.0)
    dde.set_parameters(tau=0.5)
    dde.adjust_diff()

    for t in [-0.5, 0.0, 0.01, 0.02]:
        dde.integrate(0.5 + t)

    if module_location.exists():
        module_location.unlink()


# }}}


# {{{ test_codegen_jitcdde_pickle_roundtrip


def test_codegen_jitcdde_pickle_roundtrip() -> None:
    """JiTCDDECompiledCode can be pickled and unpickled (requires module_location)."""

    pytest.importorskip("pymbolic")
    pytest.importorskip("jitcdde")

    import pickle  # noqa: S403

    module_location = pathlib.Path(tempfile.gettempdir()) / "jitcdde_orbitkit_pickle.so"

    dde = _make_dde_from_name(
        "hiv",
        "CulshawRuanWebb2003Figure44",
        symbolic=True,
        module_location=module_location,
    )
    d = dde.f.size

    # Integrate before pickling
    tau = 0.5
    dde.set_initial_conditions(np.ones(d), 0.0)
    dde.set_parameters(tau=tau)
    dde.adjust_diff()

    y_before, _, _ = dde.integrate(tau)
    y_before, _, _ = dde.integrate(tau + 0.1)

    # Round-trip
    data = pickle.dumps(dde)
    dde2 = pickle.loads(data)  # noqa: S301

    # Same metadata
    assert dde2.parameters == dde.parameters
    assert dde2.module_location == dde.module_location
    assert dde2.nlyapunov == dde.nlyapunov
    assert abs(dde2.dde.max_delay - dde.dde.max_delay) < 1.0e-14

    # Unpickled object is functional: reset, set ICs, integrate
    dde2.reset()
    dde2.set_initial_conditions(np.ones(d), 0.0)
    dde2.set_parameters(tau=tau)
    dde2.adjust_diff()

    y_after, _, _ = dde2.integrate(tau)
    y_after, _, _ = dde2.integrate(tau + 0.1)

    np.testing.assert_allclose(y_after, y_before, atol=1.0e-6)

    if module_location.exists():
        module_location.unlink()


# }}}


# {{{ test_codegen_jitcdde_pickle_no_module_location


def test_codegen_jitcdde_pickle_no_module_location() -> None:
    """Without module_location, pickling still works (Python backend fallback)."""

    pytest.importorskip("pymbolic")
    pytest.importorskip("jitcdde")

    import pickle  # noqa: S403

    dde = _make_dde_from_name(
        "hiv",
        "CulshawRuanWebb2003Figure44",
        symbolic=True,
    )
    d = dde.f.size

    # Round-trip without module_location
    data = pickle.dumps(dde)
    dde2 = pickle.loads(data)  # noqa: S301

    # Still functional (Python backend)
    tau = 0.5
    dde2.set_initial_conditions(np.ones(d), 0.0)
    dde2.set_parameters(tau=tau)
    dde2.adjust_diff()

    y, _, _ = dde2.integrate(tau)
    assert y.shape == (d,)


# }}}


# {{{ test_codegen_jitcdde_not_hashable


def test_codegen_jitcdde_not_hashable() -> None:
    """JiTCDDECompiledCode is not hashable (numpy arrays, jitcdde object)."""

    pytest.importorskip("pymbolic")
    pytest.importorskip("jitcdde")

    dde = _make_dde_from_name(
        "hiv",
        "CulshawRuanWebb2003Figure44",
        symbolic=True,
    )

    with pytest.raises(TypeError, match="unhashable"):
        hash(dde)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
