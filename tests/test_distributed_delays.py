# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import dataclass

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
    # - lambda_pif is sometimes negative, so we try to pick it if possible
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


def test_weak_gamma_homogeneous_solution() -> None:
    rng = np.random.default_rng(seed=42)
    t = np.linspace(0.0, 12.0, 512)

    # try out real solutions with at least one stable Re(lambda) < 0 root
    for alpha in [1.0, 2.0, 3.0, 4.0, 5.0]:
        for _ in range(32):
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


# {{{ test_weak_gamma_dde


@pytest.mark.parametrize("alpha", [0.5, 1.5, 2.5])
def test_weak_gamma_dde(alpha: float) -> None:
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

    # {{{ construct ODE

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


# {{{ test_uniform_homogeneous_solution


def _uniform_characteristic_equation(
    x: float, *, a: float, b: float, epsilon: float, tau: float
) -> float:
    return (
        x + a - b / (epsilon * tau * x) * np.exp(-tau * x) * np.sinh(epsilon * tau * x)
    )


def _uniform_characteristic_equation_prime(
    x: float, *, a: float, b: float, epsilon: float, tau: float
) -> float:
    tau_x = tau * x
    eps_tau_x = epsilon * tau_x
    return (
        1
        - b / x * np.exp(-tau_x) * np.cosh(eps_tau_x)
        + b / (eps_tau_x * x) * (1 + tau_x) * np.exp(-tau_x) * np.sinh(eps_tau_x)
    )


def _uniform_homogeneous_root(
    model: HomogeneousLinearModel, *, atol: float = 1.0e-8
) -> float:
    assert isinstance(model.h, sym.UniformDelayKernel)

    a = model.a
    assert isinstance(a, (int, float))
    b = model.b
    assert isinstance(b, (int, float))
    epsilon = model.h.epsilon
    assert isinstance(epsilon, (int, float))
    tau = model.h.tau
    assert isinstance(tau, (int, float))

    from functools import partial

    from scipy.optimize import root_scalar
    from scipy.special import lambertw

    # NOTE: (this needs some actual math to back it up)
    # * if a < b, the 0 branch has a chance of working
    # * if a > b, the 0 branch seems to be positive, so we start with 1
    k = 0 if a < b else 1
    while k < 10:
        result = root_scalar(
            partial(
                _uniform_characteristic_equation, a=a, b=b, epsilon=epsilon, tau=tau
            ),
            x0=np.real(lambertw(tau * b * np.exp(a * tau), k=k) / tau - a),
            fprime=partial(
                _uniform_characteristic_equation_prime,
                a=a,
                b=b,
                epsilon=epsilon,
                tau=tau,
            ),
            xtol=atol,
            rtol=atol,
        )
        assert result.converged

        if not np.iscomplex(result.root) and np.real(result.root) < 0:
            return np.real(result.root)

        k += 2

    if np.iscomplex(result.root):
        raise ValueError(f"solution is complex: lambda = {result.root}")

    if np.real(result.root) > 0:
        raise ValueError(f"solution is unstable: lambda = {result.root}")

    return np.real(result.root)


def test_uniform_homogeneous_solution() -> None:
    rng = np.random.default_rng(seed=42)
    atol = 1.0e-14

    from itertools import product

    for tau, eps in product(
        [0.5, 1.0, 2.0, 3.0, 4.0],
        [0.01, 0.1, 0.2, 0.5, 0.75, 0.9, 0.98],
    ):
        # check condition for negative roots with a > b
        h = sym.UniformDelayKernel(eps, tau)
        for _ in range(32):
            b = rng.uniform()
            a = rng.uniform(b, b + 1)
            model = HomogeneousLinearModel(a=a, b=b, h=h)

            lambda_star = _uniform_homogeneous_root(model, atol=atol)
            assert lambda_star < 0

            error_star = _uniform_characteristic_equation(
                lambda_star, a=a, b=b, epsilon=eps, tau=tau
            )
            assert abs(error_star) < atol

        # try out a positive root
        b = rng.uniform()
        a = rng.uniform(0, b)
        model = HomogeneousLinearModel(a=a, b=b, h=h)

        with pytest.raises(ValueError, match="unstable"):
            lambda_star = _uniform_homogeneous_root(model, atol=atol)

        # check condition for negative roots with a < b
        for _ in range(32):
            b = -rng.uniform()
            a = rng.uniform(-1, b)
            model = HomogeneousLinearModel(a=a, b=b, h=h)

            lambda_star = _uniform_homogeneous_root(model, atol=atol)
            assert lambda_star < 0

            error_star = _uniform_characteristic_equation(
                lambda_star, a=a, b=b, epsilon=eps, tau=tau
            )

            assert abs(error_star) < 5 * atol


# }}}

# {{{ test_uniform_dde


@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize("epsilon", [0.01, 0.1, 0.2, 0.5, 0.75, 0.9, 0.98])
# @pytest.mark.parametrize("tau", [0.5])
# @pytest.mark.parametrize("epsilon", [0.5])
def test_uniform_dde(tau: float, epsilon: float) -> None:
    from orbitkit.models import transform_distributed_delay_model

    # {{{ construct model

    rng = np.random.default_rng(seed=42)

    # NOTE: for this choice of (a, b), we should have only one negative root
    b = rng.uniform()
    a = rng.uniform(b, b + 1)

    kernel = sym.UniformDelayKernel(epsilon, tau)
    model = HomogeneousLinearModel(a=a, b=b, h=kernel)

    log.info("Model: %s", type(model))
    log.info("Equations:\n%s", model)

    ext_model = transform_distributed_delay_model(model, 1)

    log.info("Model: %s", type(ext_model))
    log.info("Equations:\n%s", ext_model)

    # }}}

    # {{{ construct DDE

    # generate code
    from orbitkit.codegen.jitcdde import JiTCDDETarget, make_input_variable

    target = JiTCDDETarget()
    source_func = target.lambdify_model(ext_model, 1)

    # compile code
    import jitcdde

    y = make_input_variable((2,))
    source = source_func(jitcdde.t, y)
    log.info("\n%s", source)

    dde = target.compile(source, y, max_delay=(1 + epsilon) * tau)

    # set initial conditions
    lambda_star = _uniform_homogeneous_root(model)
    ta, tb = (1.0 - epsilon) * tau, (1.0 + epsilon) * tau

    Y0 = rng.random()
    Z0 = (
        Y0
        * (np.exp(-ta * lambda_star) - np.exp(-tb * lambda_star))
        / ((tb - ta) * lambda_star)
    )
    log.info("lambda %.8e Y0 %g Z0 %g", lambda_star, Y0, Z0)

    dde.past_from_function(lambda t: (Y0 * np.exp(lambda_star * t), Z0))
    dde.delays = ((1.0 - epsilon) * tau, (1 + epsilon) * tau)

    # }}}

    # handle discontinuities
    dde.step_on_discontinuities()
    # dde.integrate_blindly(max_delay, step=dt)

    dt = 1.0e-4
    tspan = (0.0, 12.0)

    steps = np.arange(tspan[0], tspan[1] - dde.t, dt) + dde.t
    ts = np.empty(steps.shape, dtype=Z0.dtype)
    ys = np.empty(steps.shape, dtype=Z0.dtype)

    for i, t in enumerate(dde.t + steps):
        ts[i] = t
        ys[i], _ = dde.integrate(t)

    y_ref = Y0 * np.exp(lambda_star * ts)
    error = la.norm(ys - y_ref) / la.norm(y_ref)
    log.info("tau %.2f epsilon %.3f error %.8g", tau, epsilon, error)
    assert error < 1.0

    if not ENABLE_VISUAL:
        return

    with figure(
        TEST_DIRECTORY / f"test_dde_uniform_{tau:.2f}_{epsilon:.3f}", normalize=True
    ) as fig:
        ax = fig.gca()

        ax.plot(ts, ys)
        ax.plot(ts, y_ref, "k--")
        # ax.semilogy(ts, np.abs(ys - y_ref) + 1.0e-16)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$y$")


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
