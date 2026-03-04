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
    import jitcdde

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
) -> jitcdde.jitcdde:
    from testlib import get_model_from_module

    n = 1 if module_name == "hiv" else 2
    model = get_model_from_module(module_name, model_name, n, delayed=True)

    from orbitkit.models import transform_distributed_delay_model

    ext_model = transform_distributed_delay_model(model, n)

    from orbitkit.codegen.jitcdde import JiTCDDETarget, make_input_variable

    target = JiTCDDETarget()
    source_func = target.lambdify_model(ext_model, n)
    assert source_func is not None

    import jitcdde

    d = len(ext_model.variables)
    ys = make_input_variable(n * d)
    source = source_func(jitcdde.t, ys)

    log.info("\n%s", source)
    assert source.shape == (d * n,)

    from orbitkit.utils import tictoc

    with tictoc(f"{module_name}[{model_name}]"):
        dde = target.compile(
            source,
            ys,
            max_delay=max_delay,
            module_location=module_location,
        )

    dde.constant_past(np.ones(d * n), 0.0)
    dde.adjust_diff()

    return dde


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


def test_codegen_jitcdde_cache() -> None:
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
