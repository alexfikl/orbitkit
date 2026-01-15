# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import replace

import numpy as np
import pymbolic.primitives as prim
import pytest

from orbitkit.models import Model
from orbitkit.models.kuramoto import KuramotoAbrams
from orbitkit.typing import Array
from orbitkit.utils import get_environ_boolean, module_logger
from orbitkit.visualization import set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_boolean("ORBITKIT_ENABLE_VISUAL")

log = module_logger(__name__)
set_plotting_defaults()


# {{{ test_symbolify


def get_model_from_module(module_name: str, model_name: str, n: int) -> Model:
    # construct a dummy all-to-all connectivity matrix for the models that need it
    A = np.ones((n, n)) - np.eye(n)
    model: Model

    if module_name == "fitzhugh_nagumo":
        from orbitkit.models import fitzhugh_nagumo

        model = replace(fitzhugh_nagumo.make_model_from_name(model_name), G=A)
    elif module_name == "hiv":
        from orbitkit.models import hiv

        model = hiv.make_model_from_name(model_name)
    elif module_name == "kuramoto":
        from orbitkit.models import kuramoto

        model = kuramoto.make_model_from_name(model_name)
    elif module_name == "wang_rinzel":
        from orbitkit.models import wang_rinzel

        model = replace(wang_rinzel.make_model_from_name(model_name), A=A)
    elif module_name == "wang_buzsaki":
        from orbitkit.models import wang_buzsaki

        model = replace(wang_buzsaki.make_model_from_name(model_name), A=A)
    elif module_name == "pfeuty":
        from orbitkit.models import pfeuty

        model = replace(pfeuty.make_model_from_name(model_name), A_inh=A, A_gap=A)
    else:
        raise ValueError(f"unknown module name: '{module_name}'")

    return model


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
        # ("wilson_cowan", "")
    ],
)
def test_symbolify(module_name: str, model_name: str) -> None:
    """Check that the models can be converted to fully symbolic."""

    pytest.importorskip("pymbolic")

    n = 32

    model = get_model_from_module(module_name, model_name, n)
    args, exprs = model.symbolify(n, full=True)
    assert args[0].name == "__t"

    for i, (arg, expr) in enumerate(zip(args[1:], exprs, strict=True)):
        log.info("Eq%d:\n d%s/dt = %s", i, arg, expr)
        assert isinstance(arg, prim.Variable)

    assert all(
        arg.name == name for arg, name in zip(args[1:], model.variables, strict=True)
    )


# }}}


# {{{ test_codegen_numpy


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

    n = 32
    rng = np.random.default_rng(seed=42)

    from orbitkit.codegen.numpy import NumpyTarget

    model = get_model_from_module(module_name, model_name, n)
    d = len(model.variables)

    target = NumpyTarget()
    source = target.lambdify_model(model, n)
    assert source is not None

    ys = rng.random(d * n)
    result = source(0.0, ys)

    assert np.all(np.isfinite(result))
    assert result.shape == (d * n,)


# }}}


# {{{ test_codegen_numpy_kuramoto


def kuramoto(model: KuramotoAbrams, t: float, *thetas: Array) -> Array:
    return np.hstack([
        model.omega
        + sum(
            model.K[a, b]
            / theta_b.shape[0]
            * np.sum(
                np.sin(theta_b.reshape(-1, 1) - theta_a.reshape(1, -1) - model.alpha),
                axis=0,
            )
            for b, theta_b in enumerate(thetas)
        )
        for a, theta_a in enumerate(thetas)
    ])


@pytest.mark.parametrize("n", [32])
def test_codegen_numpy_kuramoto(n: int) -> None:
    """Check that the code gives the same result as a hand-written function."""

    pytest.importorskip("pymbolic")
    rng = np.random.default_rng(seed=42)

    from orbitkit.codegen.numpy import NumpyTarget

    model = get_model_from_module("kuramoto", "Abrams2008Figure2c", n)
    assert isinstance(model, KuramotoAbrams)

    d = len(model.variables)
    target = NumpyTarget()

    source = target.lambdify_model(model, n)
    assert source is not None

    for _ in range(8):
        ys = rng.random(d * n)
        assert np.allclose(source(0.0, ys), kuramoto(model, 0.0, ys[:n], ys[n:]))

    from orbitkit.utils import timeit

    ys = rng.random(d * n)
    result = timeit(lambda: source(0.0, ys))
    log.info("Generated: %s", result)
    result = timeit(lambda: kuramoto(model, 0.0, ys[:n], ys[n:]))
    log.info("Hardcoded: %s", result)


# }}}


# {{{ test_codegen_numpy_array_arguments


def test_codegen_numpy_array_arguments() -> None:
    """Check that the generator extracts extra arrays."""
    pytest.importorskip("pymbolic")

    from orbitkit.models.wang_buzsaki import WangBuzsaki

    n = 32
    model = get_model_from_module("wang_buzsaki", "WangBuzsaki1996Figure3a", n)
    assert isinstance(model, WangBuzsaki)
    A = model.A
    assert isinstance(A, np.ndarray)

    from orbitkit.codegen.numpy import NumpyCodeGenerator

    _, exprs = model.symbolify(n)

    cgen = NumpyCodeGenerator()
    result = cgen(exprs)
    assert result is not None

    # NOTE: the Wang-Buzsáki model has two free-floating arrays:
    #   1. The matrix `A` that was added in `get_model_from_module`
    #   2. The vector `M = sum(A, axis=1)`.
    assert len(cgen.array_arguments) == 2
    assert cgen.array_arguments.keys() == {"_arg", "_arg_0"}
    assert np.allclose(cgen.array_arguments["_arg"], A)
    assert np.allclose(cgen.array_arguments["_arg_0"], np.sum(A, axis=1))


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
