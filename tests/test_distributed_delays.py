# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as la
import pytest

import orbitkit.symbolic.primitives as sym
from orbitkit.models import Model
from orbitkit.typing import Array
from orbitkit.utils import (
    EOCRecorder,
    get_environ_boolean,
    module_logger,
    stringify_eoc,
)
from orbitkit.visualization import figure, set_plotting_defaults

if TYPE_CHECKING:
    from collections.abc import Callable

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_boolean("ORBITKIT_ENABLE_VISUAL")

log = module_logger(__name__)
set_plotting_defaults()

# {{{ linear model


def _weak_gamma_homogeneous_solution(model: HomogeneousLinearModel, t: Array) -> Array:
    lambda_star = _weak_gamma_homogeneous_root(model)
    return np.exp(lambda_star * t)


@dataclass(frozen=True)
class HomogeneousLinearModel(Model):
    a: sym.Expression
    b: sym.Expression
    h: sym.DelayKernel

    @property
    def variables(self) -> tuple[str, ...]:
        return ("u",)

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        (y,) = args

        return (-self.a * y + self.b * self.h(y),)


@dataclass(frozen=True)
class NonHomogeneousLinearModel(HomogeneousLinearModel):
    y: Callable[[Array], Array]
    g: Callable[[sym.Expression], sym.Expression]

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        (result,) = super().evaluate(t, *args)
        return (result + self.g(t),)

    def solution(self, t: Array) -> Array:
        return self.y(t)


# }}}


# {{{ test_weak_gamma_homogeneous_solution


def _weak_gamma_homogeneous_root(model: HomogeneousLinearModel) -> float:
    assert isinstance(model.h, sym.GammaDelayKernel)
    assert model.h.p == 1

    a = model.a
    assert isinstance(a, (int, float))
    b = model.b
    assert isinstance(b, (int, float))
    alpha = model.h.alpha
    assert isinstance(alpha, (int, float))

    # NOTE:
    # - these are always real for b > 0
    # - lambda_m is always negative for -alpha < a < alpha
    # - lambda_p is sometimes negative, so we try to pick it if possible
    # - existence of solutions requires that Re(alpha + lambda) > 0.

    lambda_p = -0.5 * ((alpha + a) - ((alpha - a) ** 2 + 4 * alpha * b) ** 0.5)
    lambda_m = -0.5 * ((alpha + a) + ((alpha - a) ** 2 + 4 * alpha * b) ** 0.5)
    # log.info("lambda = (%s, %s)", lambda_p, lambda_m)

    is_complex_lambda_p = np.iscomplex(lambda_p)
    is_complex_lambda_m = np.iscomplex(lambda_m)
    if is_complex_lambda_p and is_complex_lambda_m:
        raise ValueError(f"solutions are complex: lambda = ({lambda_p}, {lambda_m})")

    if not is_complex_lambda_p and -alpha < np.real(lambda_p) < 0:
        return lambda_p

    if not is_complex_lambda_m and -alpha < np.real(lambda_m) < 0:
        return lambda_m

    raise ValueError(f"solutions are unstable: lambda = ({lambda_p}, {lambda_m})")


def _weak_gamma_coefficients(
    alpha: float, rng: np.random.Generator | None = None
) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng()

    a = rng.uniform(-alpha, alpha)
    b = rng.uniform(-((alpha - a) ** 2) / (4.0 * alpha), 0)

    return a, b


@pytest.mark.parametrize("alpha", [1.0, 2.0, 3.0, 4.0, 5.0])
def test_weak_gamma_homogeneous_solution(alpha: float) -> None:
    rng = np.random.default_rng(seed=None)
    t = np.linspace(0.0, 12.0, 512)

    # try out real solutions with at least one stable Re(lambda) < 0 root
    for _ in range(12):
        a = rng.uniform(-alpha, alpha)
        b = rng.uniform(-((alpha - a) ** 2) / (4.0 * alpha), 0)
        model = HomogeneousLinearModel(a=a, b=b, h=sym.GammaDelayKernel(1.0, alpha))

        lambda_star = _weak_gamma_homogeneous_root(model)
        assert np.all(np.isfinite(np.exp(lambda_star * t)))

        error_star = lambda_star + a - alpha * b / (alpha + lambda_star)
        assert abs(error_star) < 5.0e-13

    # try out a complex solution
    a = rng.uniform(0, 1)
    b = -((alpha - a) ** 2) / (4.0 * alpha) - 0.25
    model = HomogeneousLinearModel(a=a, b=b, h=sym.GammaDelayKernel(1.0, alpha))

    with pytest.raises(ValueError, match="complex"):
        _weak_gamma_homogeneous_root(model)

    # try out no negative real part solutions
    a = -alpha - rng.uniform(0, 1)
    b = rng.uniform(-((alpha - a) ** 2) / (4.0 * alpha), a)
    model = HomogeneousLinearModel(a=a, b=b, h=sym.GammaDelayKernel(1.0, alpha))

    with pytest.raises(ValueError, match="unstable"):
        _weak_gamma_homogeneous_root(model)


# }}}


# {{{ test_distributed_delays_to_ode


@pytest.mark.parametrize("alpha", [0.5])
def test_weak_gamma_ode(alpha: float) -> None:
    from orbitkit.models import transform_distributed_delay_model

    # {{{ construct model

    rng = np.random.default_rng(seed=42)
    a, b = _weak_gamma_coefficients(alpha, rng=rng)

    kernel = sym.GammaDelayKernel(1, alpha)
    model = HomogeneousLinearModel(a=a, b=b, h=kernel)

    log.info("Model: %s", type(model))
    log.info("Equations:\n%s", model)

    ext_model = transform_distributed_delay_model(model, 1)

    log.info("Model: %s", type(ext_model))
    log.info("Equations:\n%s", ext_model)

    # }}}

    # {{{ construct solution

    from orbitkit.codegen.numpy import NumpyTarget

    lambda_star = _weak_gamma_homogeneous_root(model)
    y00 = rng.random()
    y0 = np.array([
        y00,
        y00 * alpha / (alpha + lambda_star),
    ])

    target = NumpyTarget()
    source = target.lambdify_model(ext_model, 1)

    # }}}

    # {{{ solve

    from scipy.integrate import solve_ivp

    eoc = EOCRecorder()
    tspan = (0.0, 10.0)

    for atol in [1.0e-5, 1.0e-7, 1.0e-9, 1.0e-11]:
        result = solve_ivp(
            source,
            tspan,
            y0,
            method="RK45",
            atol=atol,
            rtol=atol,
        )

        y_ref = y0[0] * np.exp(lambda_star * result.t)
        error = la.norm(result.y[0] - y_ref) / la.norm(y_ref)
        log.info("tol %.8e error: %.8e", atol, error)

        eoc.add_data_point(np.max(np.diff(result.t)), error)

    log.info("EOC:\n%s", stringify_eoc(eoc))
    assert eoc.estimated_order > 4.0

    # }}}

    if not ENABLE_VISUAL:
        return

    with figure(
        TEST_DIRECTORY / f"test_dde_weak_gamma_{alpha:.2f}", normalize=True
    ) as fig:
        ax = fig.gca()

        # ax.plot(result.t, result.y[0])
        # ax.plot(result.t, y_ref, "k--")
        ax.semilogy(result.t, np.abs(result.y[0] - y_ref) + 1.0e-16)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$y$")


# }}}


# {{{ test_distributed_delays_to_dde

# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
