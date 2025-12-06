# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import numpy as np
from pymbolic.mapper import IdentityMapper as IdentityMapperBase
from pymbolic.mapper import WalkMapper as WalkMapperBase
from pymbolic.mapper.flattener import FlattenMapper as FlattenMapperBase
from pymbolic.mapper.stringifier import PREC_NONE
from pymbolic.mapper.stringifier import StringifyMapper as StringifyMapperBase
from pymbolic.typing import Expression as PymbolicExpression

import orbitkit.symbolic.primitives as sym
from orbitkit.typing import Array

# {{{ IdentityMapper


class IdentityMapper(IdentityMapperBase[[]]):
    def map_contract(self, expr: sym.Contract, /) -> PymbolicExpression:
        aggregate = self.rec_arith(expr.aggregate)
        if aggregate is expr.aggregate:
            return expr

        return type(expr)(aggregate=aggregate, axes=expr.axes)

    def map_einstein_summation(
        self, expr: sym.EinsteinSummation, /
    ) -> PymbolicExpression:
        operands = tuple(self.rec_arith(operand) for operand in expr.operands)
        if all(a is b for a, b in zip(operands, expr.operands, strict=True)):
            return expr

        return type(expr)(subscripts=expr.subscripts, operands=operands)

    def map_dot_product(self, expr: sym.DotProduct, /) -> PymbolicExpression:
        # NOTE: mypy is upset because left/right can also be ndarrays
        left = self.rec_arith(expr.left)  # type: ignore[arg-type]
        right = self.rec_arith(expr.right)  # type: ignore[arg-type]
        if left is expr.left and right is expr.right:
            return expr

        return type(expr)(left, right)

    def map_reshape(self, expr: sym.Reshape, /) -> PymbolicExpression:
        aggregate = self.rec_arith(expr.aggregate)
        if aggregate is expr.aggregate:
            return expr

        return type(expr)(aggregate=aggregate, shape=expr.shape)

    def map_variable_with_delay(
        self, expr: sym.VariableWithDelay, /
    ) -> PymbolicExpression:
        tau = self.rec_arith(expr.tau)
        if tau is expr.tau:
            return expr

        return type(expr)(expr.name, tau)

    def map_dirac_delay_kernel(
        self, expr: sym.DiracDelayKernel, /
    ) -> PymbolicExpression:
        return type(expr)(tau=self.rec_arith(expr.tau))

    def map_uniform_delay_kernel(
        self, expr: sym.UniformDelayKernel, /
    ) -> PymbolicExpression:
        return type(expr)(
            epsilon=self.rec_arith(expr.epsilon),
            tau=self.rec_arith(expr.tau),
        )

    def map_triangular_delay_kernel(
        self, expr: sym.TriangularDelayKernel, /
    ) -> PymbolicExpression:
        return type(expr)(
            epsilon=self.rec_arith(expr.epsilon),
            tau=self.rec_arith(expr.tau),
        )

    def map_gamma_delay_kernel(
        self, expr: sym.GammaDelayKernel, /
    ) -> PymbolicExpression:
        return type(expr)(p=expr.p, alpha=self.rec_arith(expr.alpha))


# }}}


# {{{ WalkMapper


class WalkMapper(WalkMapperBase[[]]):
    def map_contract(self, expr: sym.Contract, /) -> None:
        if not self.visit(expr):
            return

        self.rec(expr.aggregate)
        self.post_visit(expr)

    def map_einstein_summation(self, expr: sym.EinsteinSummation, /) -> None:
        if not self.visit(expr):
            return

        for operand in expr.operands:
            self.rec(operand)
        self.post_visit(expr)

    def map_dot_product(self, expr: sym.DotProduct, /) -> None:
        if not self.visit(expr):
            return

        self.rec(expr.left)  # type: ignore[arg-type]
        self.rec(expr.right)  # type: ignore[arg-type]
        self.post_visit(expr)

    def map_variable_with_delay(self, expr: sym.VariableWithDelay, /) -> None:
        if not self.visit(expr):
            return

        self.rec(expr.tau)
        self.post_visit(expr)

    def map_delay_kernel(self, expr: sym.DelayKernel) -> None:
        if not self.visit(expr):
            return
        self.post_visit(expr)


# }}}


# {{{ StringifyMapper


class StringifyMapper(StringifyMapperBase[Any]):
    def map_variable(self, expr: sym.Variable, /, enclosing_prec: int) -> str:  # noqa: PLR6301
        from sympy.printing.pretty.pretty_symbology import pretty_symbol

        result = pretty_symbol(expr.name)
        return str(result)

    def map_variable_with_delay(
        self, expr: sym.VariableWithDelay, /, enclosing_prec: int
    ) -> str:
        from sympy.printing.pretty.pretty_symbology import pretty_symbol

        name = pretty_symbol(expr.name)
        tau = self.rec(expr.tau, PREC_NONE)

        return f"{name}(·-({tau}))"

    def map_numpy_array(  # noqa: PLR6301
        self, expr: np.ndarray[tuple[int, ...], np.dtype[Any]], /, enclosing_prec: int
    ) -> str:
        return repr(expr)

    def map_contract(self, expr: sym.Contract, /, enclosing_prec: int) -> str:
        aggregate = self.rec(expr.aggregate, PREC_NONE)
        return f"sum({aggregate}, axis={expr.axes})"

    def map_reshape(self, expr: sym.Reshape, /, enclosing_prec: int) -> str:
        aggregate = self.rec(expr.aggregate, PREC_NONE)
        return f"({aggregate}).reshape{expr.shape}"

    def map_dot_product(self, expr: sym.DotProduct, /, enclosing_prec: int) -> str:
        left = self.rec(expr.left, PREC_NONE)  # type: ignore[arg-type]
        right = self.rec(expr.right, PREC_NONE)  # type: ignore[arg-type]
        return f"dot({left}, {right})"

    def map_delay_kernel(self, expr: sym.DelayKernel, /, enclosing_prec: int) -> str:
        from dataclasses import fields

        params = ", ".join(
            f"{self.rec(getattr(expr, f.name), enclosing_prec)}" for f in fields(expr)
        )
        return f"{type(expr).__name__}(·; {params})"


def stringify(expr: sym.Expression | Array) -> str:
    return StringifyMapper()(expr)  # type: ignore[arg-type]


# }}}


# {{{ FattenMapper


class FlattenMapper(FlattenMapperBase, IdentityMapper):
    pass


def flatten(expr: sym.Expression | Array) -> sym.Expression | Array:
    return FlattenMapper()(expr)  # type: ignore[arg-type,return-value]


# }}}
