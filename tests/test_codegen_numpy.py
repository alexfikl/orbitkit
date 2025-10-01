# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pymbolic.primitives as prim
import pytest

from orbitkit.models.kuramoto import KuramotoAbrams
from orbitkit.models.symbolic import Model
from orbitkit.typing import Array
from orbitkit.utils import get_environ_boolean, module_logger, set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_boolean("ORBITKIT_ENABLE_VISUAL")

log = module_logger(__name__)
set_plotting_defaults()


# {{{ test_symbolify


def get_model_from_module(module_name: str, model_name: str) -> Model:
    if module_name == "kuramoto":
        import orbitkit.models.kuramoto as module
    else:
        raise ValueError(f"unknown module name: '{module_name}'")

    return module.make_model_from_name(model_name)


@pytest.mark.parametrize(
    ("module_name", "model_name"),
    [
        ("kuramoto", "Abrams2008Figure2a"),
    ],
)
def test_symbolify(module_name: str, model_name: str) -> None:
    n = 32

    model = get_model_from_module(module_name, model_name)
    args, exprs = model.symbolify(n)
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
        ("kuramoto", "Abrams2008Figure2a"),
    ],
)
def test_codegen_numpy(module_name: str, model_name: str) -> None:
    n = 32
    rng = np.random.default_rng(seed=42)

    from orbitkit.models.targets import NumpyTarget

    model = get_model_from_module(module_name, model_name)
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
                theta_b.reshape(-1, 1) - theta_a.reshape(1, -1) - model.alpha,
                axis=0,
            )
            for b, theta_b in enumerate(thetas)
        )
        for a, theta_a in enumerate(thetas)
    ])


@pytest.mark.parametrize("n", [32])
def test_codegen_numpy_kuramoto(n: int) -> None:
    rng = np.random.default_rng(seed=42)

    from orbitkit.models.targets import NumpyTarget

    model = get_model_from_module("kuramoto", "Abrams2008Figure2c")
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
