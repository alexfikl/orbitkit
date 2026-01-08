# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
from pymbolic.mapper import IdentityMapper as IdentityMapperBase
from pymbolic.mapper import WalkMapper as WalkMapperBase
from pymbolic.mapper.flattener import FlattenMapper as FlattenMapperBase
from pymbolic.mapper.stringifier import PREC_NONE
from pymbolic.mapper.stringifier import StringifyMapper as StringifyMapperBase
from pymbolic.typing import Expression as PymbolicExpression

import orbitkit.symbolic.primitives as sym
from orbitkit.typing import Array

if TYPE_CHECKING:
    from collections.abc import Mapping


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
        # NOTE: ty is upset because left/right can also be ndarrays
        left = self.rec_arith(expr.left)  # ty: ignore[invalid-argument-type]
        right = self.rec_arith(expr.right)  # ty: ignore[invalid-argument-type]
        if left is expr.left and right is expr.right:
            return expr

        return type(expr)(left, right)

    def map_reshape(self, expr: sym.Reshape, /) -> PymbolicExpression:
        aggregate = self.rec_arith(expr.aggregate)
        if aggregate is expr.aggregate:
            return expr

        return type(expr)(aggregate=aggregate, shape=expr.shape)

    def map_call_delay(self, expr: sym.CallDelay, /) -> PymbolicExpression:
        aggregate = self.rec_arith(expr.aggregate)
        tau = self.rec_arith(expr.tau)
        if aggregate is expr.aggregate and tau is expr.tau:
            return expr

        return type(expr)(aggregate, tau)

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

        self.rec(expr.left)  # ty: ignore[invalid-argument-type]
        self.rec(expr.right)  # ty: ignore[invalid-argument-type]
        self.post_visit(expr)

    def map_call_delay(self, expr: sym.CallDelay, /) -> None:
        if not self.visit(expr):
            return

        self.rec(expr.aggregate)
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

    def map_call_delay(self, expr: sym.CallDelay, /, enclosing_prec: int) -> str:
        aggregate = self.rec(expr.aggregate, PREC_NONE)
        tau = self.rec(expr.tau, PREC_NONE)

        return f"{aggregate}(·-({tau}))"

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
        left = self.rec(expr.left, PREC_NONE)
        right = self.rec(expr.right, PREC_NONE)
        return f"dot({left}, {right})"

    def map_delay_kernel(self, expr: sym.DelayKernel, /, enclosing_prec: int) -> str:
        from dataclasses import fields

        params = ", ".join(
            f"{self.rec(getattr(expr, f.name), enclosing_prec)}" for f in fields(expr)
        )
        return f"{type(expr).__name__}(·; {params})"


def stringify(expr: sym.Expression | Array) -> str:
    return StringifyMapper()(expr)


# }}}


# {{{ FattenMapper


class FlattenMapper(FlattenMapperBase, IdentityMapper):
    pass


def flatten(expr: sym.Expression | Array) -> sym.Expression | Array:
    return FlattenMapper()(expr)  # ty: ignore[invalid-argument-type,invalid-return-type]


# }}}


# {{{ RenameMapper


class RenameMapper(IdentityMapper):
    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.mapping = mapping

    def map_variable(self, expr: sym.Variable) -> sym.Expression:
        new_name = self.mapping.get(expr.name)
        if new_name is None:
            return expr

        if new_name == expr.name:
            return expr

        return replace(expr, name=new_name)


def rename_variables(
    expr: tuple[sym.Expression, ...], mapping: Mapping[str, str]
) -> tuple[sym.Expression, ...]:
    result = RenameMapper(mapping)(expr)
    assert isinstance(result, tuple)

    return result  # ty: ignore[invalid-return-type]


# }}}
