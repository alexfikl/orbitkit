# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

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
        self.variables = set()
        self.kernels = set()

    def visit(self, expr: sym.Expression) -> bool:
        if isinstance(expr, sym.VariableWithDelay):
            self.variables.add(expr)
        elif isinstance(expr, sym.DelayKernel):
            self.kernels.add(expr)
        else:
            return True

        return False


@pytest.mark.parametrize(
    "knl",
    [
        sym.DiracDelayKernel(sym.var("tau") + 1),
        sym.UniformDelayKernel(sym.var("epsilon"), 1.0),
        sym.TriangularDelayKernel(sym.var("epsilon"), 1.0),
        sym.GammaDelayKernel(1, sym.var("alpha")),
        sym.GammaDelayKernel(2, sym.var("alpha")),
        sym.GammaDelayKernel(3, sym.var("alpha")),
        sym.GammaDelayKernel(7, sym.var("alpha")),
    ],
)
def test_linear_chain_trick(knl: sym.DelayKernel) -> None:
    from orbitkit.models.rate_functions import SigmoidRate

    s = SigmoidRate(1, 0, sym.var("sigma"))

    y = sym.var("y")
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
        log.info("%4s: %s", stringify(sym.var(name)), stringify(flatten(eq)))


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
