# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import orbitkit.symbolic.primitives as sym
from orbitkit.symbolic.mappers import WalkMapper
from orbitkit.utils import get_environ_boolean, module_logger
from orbitkit.visualization import set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_boolean("ORBITKIT_ENABLE_VISUAL")

log = module_logger(__name__)
set_plotting_defaults()


# {{{ test_linear_chain_trick


class DelayFinder(WalkMapper):
    def __init__(self) -> None:
        self.variables: set[sym.CallDelay] = set()
        self.kernels: set[sym.DelayKernel] = set()

    def visit(self, expr: object) -> bool:
        if isinstance(expr, sym.CallDelay):
            self.variables.add(expr)
        elif isinstance(expr, sym.DelayKernel):
            self.kernels.add(expr)
        else:
            return True

        return False


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

    log.info("\n")
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
    from orbitkit.models.wilson_cowan import WilsonCowan1, WilsonCowanParameter

    rng = np.random.default_rng(seed=42)

    n = 10
    s = SigmoidRate(1, 0, sym.Variable("sigma"))
    Ep = WilsonCowanParameter(
        sigmoid=s,
        kernels=(knl, knl),
        weights=(rng.random((n, n)), rng.random((10, 10))),
        forcing=rng.random(10),
    )
    Ip = WilsonCowanParameter(
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

    log.info("")

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

    if not ENABLE_VISUAL:
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
