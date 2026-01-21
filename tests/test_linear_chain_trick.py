# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pymbolic.primitives as prim
import pytest

import orbitkit.symbolic.primitives as sym
from orbitkit.symbolic.mappers import WalkMapper
from orbitkit.utils import enable_test_plotting, module_logger
from orbitkit.visualization import set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)
set_plotting_defaults()


# {{{ test_dirac_delay_distributor


def test_dirac_delay_distributor() -> None:
    tau = sym.Variable("tau")
    t = sym.Variable("t")
    x = [sym.Variable("x"), sym.Variable("y")]

    a = sym.Variable("a")
    b = sym.Variable("b")

    def dirac(y: sym.Expression) -> sym.Expression:
        return sym.DiracDelayKernel(tau)(y)

    from orbitkit.models.linear_chain_tricks import DiracDelayDistributor

    delay = DiracDelayDistributor(tau, time=t, inputs=x)

    # check that "constants" just get ignored
    expr = a + b
    assert expr == delay(expr)

    # check that time gets expanded
    expr = a + b * t
    assert delay(expr) == a + b * (t - tau)

    expr = a + b * x[0] * x[1]
    assert delay(expr) == a + b * dirac(x[0]) * dirac(x[1])

    # check calls
    expr = a * sym.sin(x[1])
    assert delay(expr) == a * sym.sin(dirac(x[1]))

    # check no nesting
    expr = a + b * (sym.sin(sym.UniformDelayKernel(0.5, 1.0)(x[0])))
    with pytest.raises(ValueError, match="cannot distribute"):
        delay(expr)

    # check that time and inputs are ignored if not provided
    delay = DiracDelayDistributor(tau)

    expr = a + b * t
    assert delay(expr) == dirac(a) + dirac(b) * dirac(t)

    expr = t + x[0] / x[1]
    assert delay(expr) == dirac(t) + dirac(x[0]) / dirac(x[1])


# }}}


# {{{ test_linear_chain_trick


class DelayFinder(WalkMapper):
    def __init__(self) -> None:
        self.variables: set[prim.Call] = set()
        self.kernels: set[prim.Call] = set()

    def visit(self, expr: object) -> bool:
        if not isinstance(expr, prim.Call):
            return True

        func = expr.function
        if isinstance(func, sym.DiracDelayKernel):
            self.variables.add(expr)
            return False
        elif isinstance(func, sym.DelayKernel):
            self.kernels.add(expr)
            return False
        else:
            return True


@pytest.mark.parametrize(
    "knl",
    [
        sym.DiracDelayKernel(sym.Variable("tau") + 1),
        sym.UniformDelayKernel(sym.Variable("epsilon"), 1.5),
        sym.TriangularDelayKernel(sym.Variable("epsilon"), 1.5),
        sym.GammaDelayKernel(1, sym.Variable("alpha")),
        sym.GammaDelayKernel(2, sym.Variable("alpha")),
        sym.GammaDelayKernel(3, sym.Variable("alpha")),
        sym.GammaDelayKernel(7, sym.Variable("alpha")),
    ],
)
def test_linear_chain_trick(knl: sym.DelayKernel) -> None:
    from orbitkit.models.rate_functions import SigmoidRate

    s = SigmoidRate(1, 0, sym.Variable("sigma"))

    y = sym.Variable("y")
    expr = -y + s(knl(y))

    from orbitkit.models.linear_chain_tricks import transform_delay_kernels

    result, equations = transform_delay_kernels(expr)
    assert result is not None
    assert isinstance(equations, dict)

    # check that the remaining variables / kernels match expectations
    finder = DelayFinder()
    finder(result)
    for eq in equations.values():
        finder(eq)

    assert not finder.kernels
    if isinstance(knl, sym.DiracDelayKernel):
        assert len(equations) == 0
        assert len(finder.variables) == 1
    elif isinstance(knl, sym.UniformDelayKernel):
        assert len(equations) == 1
        assert len(finder.variables) == 2
    elif isinstance(knl, sym.TriangularDelayKernel):
        assert len(equations) == 2
        assert len(finder.variables) == 3
    elif isinstance(knl, sym.GammaDelayKernel):
        assert len(equations) == knl.p
        assert len(finder.variables) == 0
    else:
        raise TypeError(f"unknown kernel type: {type(knl)}")

    from orbitkit.symbolic.mappers import flatten, stringify

    log.info("%4s: %s", stringify(y), stringify(flatten(expr)))
    log.info("%4s: %s", stringify(y), stringify(flatten(result)))

    for name, eq in equations.items():
        log.info("%4s: %s", stringify(sym.Variable(name)), stringify(flatten(eq)))


# }}}


# {{{ test_wilson_cowan_linear_chain_tricks


@pytest.mark.parametrize(
    "knl",
    [
        sym.DiracDelayKernel(sym.Variable("tau") + 1),
        sym.UniformDelayKernel(sym.Variable("epsilon"), 1.5),
        sym.TriangularDelayKernel(sym.Variable("epsilon"), 1.5),
        sym.GammaDelayKernel(1, sym.Variable("alpha")),
        sym.GammaDelayKernel(2, sym.Variable("alpha")),
        sym.GammaDelayKernel(3, sym.Variable("alpha")),
        sym.GammaDelayKernel(7, sym.Variable("alpha")),
    ],
)
def test_wilson_cowan_linear_chain_tricks(knl: sym.DelayKernel) -> None:
    from orbitkit.models.rate_functions import SigmoidRate
    from orbitkit.models.wilson_cowan import WilsonCowan1, WilsonCowanPopulation

    rng = np.random.default_rng(seed=42)

    n = 10
    s = SigmoidRate(1, 0, sym.Variable("sigma"))
    Ep = WilsonCowanPopulation(
        sigmoid=s,
        kernels=(knl, knl),
        weights=(rng.random((n, n)), rng.random((10, 10))),
        forcing=rng.random(10),
    )
    Ip = WilsonCowanPopulation(
        sigmoid=s,
        kernels=(knl, knl),
        weights=(rng.random((n, n)), rng.random((10, 10))),
        forcing=rng.random(10),
    )

    model = WilsonCowan1(E=Ep, I=Ip)
    log.info("Model:\n%s", model)

    from orbitkit.models.linear_chain_tricks import transform_delay_kernels

    args, exprs = model.symbolify(n, full=True)
    result, eqs = transform_delay_kernels(exprs)
    assert len(exprs) == len(result)

    from orbitkit.symbolic.mappers import stringify

    lines = []
    for i, (y, eq) in enumerate(zip(args[1:], result, strict=True)):
        lines.append(f"[{i:02d}]:\n\td{stringify(y)}/dt = {stringify(eq)}")

    for i, (name, eq) in enumerate(eqs.items()):
        i += len(exprs)  # noqa: PLW2901
        y = sym.Variable(name)
        lines.append(f"[{i:02d}]:\n\td{stringify(y)}/dt = {stringify(eq)}")

    log.info("Model:\n%s", "\n".join(lines))

    # check that the remaining variables / kernels match expectations
    finder = DelayFinder()
    finder(result)
    for eq in eqs.values():
        finder(eq)

    assert not finder.kernels
    if isinstance(knl, sym.DiracDelayKernel):
        assert len(eqs) == 0
        assert len(finder.variables) == 1 * 2
    elif isinstance(knl, sym.UniformDelayKernel):
        assert len(eqs) == 1 * 2
        assert len(finder.variables) == 2 * 2
    elif isinstance(knl, sym.TriangularDelayKernel):
        assert len(eqs) == 2 * 2
        assert len(finder.variables) == 3 * 2
    elif isinstance(knl, sym.GammaDelayKernel):
        assert len(eqs) == knl.p * 2
        assert len(finder.variables) == 0
    else:
        raise TypeError(f"unknown kernel type: {type(knl)}")


# }}}


# {{{ test_sum_of_exponentials


@pytest.mark.parametrize("method", ["varpo", "mpm"])
@pytest.mark.parametrize(
    ("p", "alpha"),
    [
        (1.0, np.pi),
        (2.0, np.pi),
        (3.0, np.pi),
        (np.pi, np.pi),
        (1.5, 1.5),
        # NOTE: these are a bit slow on the `varpo` method
        (13.5, 1.5),
        # (23, 0.5),
    ],
)
def test_sum_of_exponentials(method: str, p: float, alpha: float) -> None:
    from orbitkit.models.linear_chain_tricks import (
        optimal_soe_gamma_points,
        soe_gamma_mpm,
        soe_gamma_varpo,
    )

    soe_eps = 1.0e-8
    t = optimal_soe_gamma_points(p, alpha, rtol=soe_eps)

    if method == "varpo":
        ws, lambdas = soe_gamma_varpo(t, p, alpha, atol=soe_eps)
    elif method == "mpm":
        ws, lambdas = soe_gamma_mpm(t, p, alpha, atol=soe_eps)
    else:
        raise ValueError(f"unknown method: {method!r}")

    log.info("(%g, %g): t in [%g, %g] dt %g", p, alpha, t[0], t[-1], t[1] - t[0])

    from scipy.special import gamma

    # check approximation at given t
    y_approx = np.real_if_close(np.exp(lambdas[None, :] * t[:, None]) @ ws)
    y_ref = alpha**p / gamma(p) * t ** (p - 1) * np.exp(-alpha * t)

    error = np.linalg.norm(y_approx - y_ref) / np.linalg.norm(y_ref)
    log.info("(%g, %g): size %d error %.8e", p, alpha, ws.size, error)

    if method == "varpo":
        assert error < 1.0
    elif method == "mpm":
        if p == 3.0:
            # NOTE: seems to only happen on Python 3.10 on the CI
            assert error < 200.0 * soe_eps
        else:
            assert error < 80.0 * soe_eps
    else:
        raise AssertionError

    # check error at more t
    t = optimal_soe_gamma_points(p, alpha, 0.0, t[-1], dt=(t[1] - t[0]) / 3)
    y_approx = np.real_if_close(np.exp(lambdas[None, :] * t[:, None]) @ ws)
    y_ref = alpha**p / gamma(p) * t ** (p - 1) * np.exp(-alpha * t)

    error = np.linalg.norm(y_approx - y_ref) / np.linalg.norm(y_ref)
    log.info("(%g, %g): size %d error %.8e", p, alpha, ws.size, error)

    if method == "varpo":
        assert error < 1.0
    elif method == "mpm":
        if p == 1.5:
            # FIXME: not sure why this happens?
            assert error < 2.0e-3
        else:
            assert error < 80.0 * soe_eps
    else:
        raise AssertionError

    if not enable_test_plotting():
        return

    from orbitkit.visualization import figure

    with figure(
        TEST_DIRECTORY / f"test_sum_of_exp_{p:05.2f}_{alpha:02.2f}", normalize=True
    ) as fig:
        ax = fig.gca()

        ax.plot(t, y_approx)
        ax.plot(t, y_ref, "k--")
        ax.set_xlabel("$t$")
        ax.set_ylabel(rf"$\mathrm{{Gamma}}(t; p={p:.2f}, \alpha={alpha:.2f})$")

    with figure(
        TEST_DIRECTORY / f"test_sum_of_exp_{p:05.2f}_{alpha:02.2f}_weights",
        normalize=True,
    ) as fig:
        ax = fig.gca()

        ax.plot(ws.real)
        ax.plot(ws.imag)
        ax.set_xlabel("$k$")
        ax.set_ylabel("$w_k$")


# }}}


# {{{ test_pade_gamma


@pytest.mark.parametrize(
    ("p", "alpha"),
    [
        # NOTE: these should just work great
        (1.0, np.pi),
        (2.0, np.pi),
        (3.0, np.pi),
        # NOTE: these are more complicated
        (np.pi, np.pi),
        (1.5, 1.5),
        (13.5, 1.5),
        (31.5, 1.5),
    ],
)
@pytest.mark.parametrize(
    ("n", "m"),
    [
        (1, 3),
        (1, 6),
        (1, 9),
        (2, 3),
        (5, 6),
        (8, 9),
    ],
)
def test_pade_gamma(p: float, alpha: float, n: int, m: int) -> None:
    from orbitkit.models.linear_chain_tricks import pade_gamma

    pcoeff, qcoeff = pade_gamma(p, alpha, n=n, m=m)
    ppoly = np.polynomial.Polynomial(pcoeff)
    qpoly = np.polynomial.Polynomial(qcoeff)

    # test accuracy on [0, alpha]
    s = np.linspace(0, alpha, 256)
    gamma_ref = (alpha / (s + alpha)) ** p
    gamma_approx = ppoly(s) / qpoly(s)

    error = np.linalg.norm(gamma_approx - gamma_ref) / np.linalg.norm(gamma_ref)
    log.info("Error[%g, %g, %d, %d]: %.8e", p, alpha, n, m, error)
    assert error < 1.0

    # test accuracy on [alpha, 5 alpha]
    s = np.linspace(alpha, 5.0 * alpha, s.size)
    gamma_ref = (alpha / (s + alpha)) ** p
    gamma_approx = ppoly(s) / qpoly(s)

    error = np.linalg.norm(gamma_approx - gamma_ref) / np.linalg.norm(gamma_ref)
    log.info("Error[%g, %g, %d, %d]: %.8e", p, alpha, n, m, error)
    if p < 10.0:
        assert error < 1.0

    if not enable_test_plotting():
        return

    from orbitkit.visualization import figure

    with figure(
        TEST_DIRECTORY / f"test_pade_gamma_{p:05.2f}_{alpha:02.2f}_{n}_{m}",
        normalize=True,
    ) as fig:
        ax = fig.gca()

        ax.plot(s, gamma_approx)
        ax.plot(s, gamma_ref, "k--")
        ax.set_xlabel("$t$")
        ax.set_ylabel(rf"$\mathrm{{Gamma}}(t; p={p:.2f}, \alpha={alpha:.2f})$")


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
