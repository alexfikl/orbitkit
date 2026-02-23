# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pymbolic.primitives as prim
import pytest

from orbitkit.models.kuramoto import KuramotoAbrams
from orbitkit.typing import Array
from orbitkit.utils import module_logger
from orbitkit.visualization import set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)
set_plotting_defaults()


# {{{ test_symbolify


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
def test_symbolify(module_name: str, model_name: str) -> None:
    """Check that the models can be converted to fully symbolic."""

    pytest.importorskip("pymbolic")
    from testlib import get_model_from_module

    n = 32
    model = get_model_from_module(module_name, model_name, n)
    args, exprs = model.symbolify(n, full=True)
    assert args[0].name == "t"

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
    from testlib import get_model_from_module

    n = 32
    rng = np.random.default_rng(seed=42)

    model = get_model_from_module(module_name, model_name, n)
    d = len(model.variables)

    from orbitkit.codegen.numpy import NumpyTarget

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
    from testlib import get_model_from_module

    rng = np.random.default_rng(seed=42)

    model = get_model_from_module("kuramoto", "Abrams2008Figure2c", n)
    d = len(model.variables)
    assert isinstance(model, KuramotoAbrams)

    from orbitkit.codegen.numpy import NumpyTarget

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

    from testlib import get_model_from_module

    from orbitkit.models.wang_buzsaki import WangBuzsaki

    n = 32
    model = get_model_from_module("wang_buzsaki", "WangBuzsaki1996Figure3a", n)
    assert isinstance(model, WangBuzsaki)
    A = model.A
    assert isinstance(A, np.ndarray)

    from orbitkit.codegen.numpy import NumpyCodeGenerator

    inputs, exprs = model.symbolify(n)

    cgen = NumpyCodeGenerator(inputs={inp.name for inp in inputs})
    result = cgen(exprs)
    assert result is not None

    # NOTE: the Wang-Buzsáki model has two free-floating arrays:
    #   1. The connection matrix `A`,
    #   2. The vector `M = sum(A, axis=1)`.
    assert len(cgen.array_arguments) == 2
    assert cgen.array_arguments.keys() == {"_arg", "_arg_0"}
    assert np.allclose(cgen.array_arguments["_arg"], A)
    assert np.allclose(cgen.array_arguments["_arg_0"], np.sum(A, axis=1))
    assert len(cgen.parameters) == 0


# }}}

# {{{ test_codegen_numpy_parameters


def test_codegen_numpy_parameters() -> None:
    """Check parameter gathering."""
    pytest.importorskip("pymbolic")

    import orbitkit.symbolic.primitives as sym

    n = 8
    A = np.ones(n) - np.eye(n)
    x = sym.Variable("x")
    eps = sym.Variable("eps")

    from orbitkit.codegen.numpy import NumpyCodeGenerator

    # {{{ check explicit parameter

    cgen = NumpyCodeGenerator(inputs={x.name})
    expr = -x + sym.exp(x + sym.DotProduct(sym.Product((eps, A)), x))  # ty: ignore[invalid-argument-type]
    result = cgen(expr)

    assert len(cgen.array_arguments) == 1
    assert len(cgen.object_array_arguments) == 0
    assert len(cgen.parameters) == 1

    # }}}

    # {{{ check parameter hidden in object array due to eager `eps * A`

    cgen = NumpyCodeGenerator(inputs={x.name})
    expr = -x + sym.exp(x + sym.DotProduct(eps * A, x))  # ty: ignore[unsupported-operator]
    result = cgen(expr)

    # NOTE: we have eps as a parameter, but it is not caught at this level, because
    # it's hidden in the object array. We should still be able to gather it later.
    assert len(cgen.array_arguments) == 0
    assert len(cgen.object_array_arguments) == 1
    assert len(cgen.parameters) == 0

    from orbitkit.codegen.numpy import NumpyTarget

    target = NumpyTarget()
    code = target.generate_code((x,), (expr,))
    log.info("Source:\n%s", code.source)

    func = target.lambdify(code, parameters={"eps": 1})

    x_ref = np.ones(n)
    result = func(x_ref)
    assert result.shape == (n,)
    assert np.all(np.isclose(result, -x_ref + np.exp(x_ref + A @ x_ref)))

    # }}}


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
