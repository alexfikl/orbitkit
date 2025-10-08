# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import symengine as sp
from pymbolic import primitives as prim
from pymbolic.interop.symengine import PymbolicToSymEngineMapper

import orbitkit.models.symbolic as sym
from orbitkit.utils import module_logger

log = module_logger(__name__)

SymEngineExpression: TypeAlias = sp.Basic | np.ndarray[tuple[int, ...], np.dtype[Any]]


class SymEngineMapper(PymbolicToSymEngineMapper):
    input_map: dict[str, SymEngineExpression]

    def __init__(self, input_map: dict[str, SymEngineExpression]) -> None:
        super().__init__()
        self.input_map = input_map

    def map_call(self, expr: prim.Call) -> SymEngineExpression:
        if isinstance(expr.function, prim.Variable):
            name = expr.function.name
            try:
                func = getattr(self.sym.functions, name)
            except AttributeError:
                func = self.sym.Function(name)

            from pytools.obj_array import vectorize_n_args

            return vectorize_n_args(func, *[self.rec(par) for par in expr.parameters])  # type: ignore[no-untyped-call]
        else:
            raise NotImplementedError(expr)

    def map_matrix_symbol(self, expr: sym.MatrixSymbol) -> SymEngineExpression:
        if expr.name in self.input_map:
            result = self.input_map[expr.name]
            assert expr.shape == result.shape
        else:
            result = super().map_variable(expr)  # type: ignore[no-untyped-call]

        return result

    def map_contract(self, expr: sym.Contract) -> SymEngineExpression:
        aggregate = self.rec(expr.aggregate)
        return np.sum(aggregate, axis=expr.axes)

    def map_reshape(self, expr: sym.Reshape) -> SymEngineExpression:
        aggregate = self.rec(expr.aggregate)
        return np.reshape(aggregate, shape=expr.shape)

    def map_dot_product(self, expr: sym.DotProduct) -> SymEngineExpression:
        left = self.rec(expr.left)  # type: ignore[arg-type]
        right = self.rec(expr.right)  # type: ignore[arg-type]

        return np.dot(left, right)
