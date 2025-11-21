# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pytools
from pymbolic.mapper.stringifier import PREC_NONE, StringifyMapper

import orbitkit.models.symbolic as sym
from orbitkit.codegen import Code, Target, execute_code
from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)

# TODO: will need to make this into a proper compiler with statements and
# assignments and whatnot at some point. The main driving force for that is
# the need for CSE to save intermediate results..

# {{{ numpy code generator


@dataclass(frozen=True)
class NumpyCodeGenerator(StringifyMapper[Any]):
    """A code generator that stringifies a symbolic :mod:`pymbolic` expression."""

    module: str = "np"
    """Name of the :mod:`numpy` module. This can be changed for :mod:`numpy`
    compatible module with some work (e.g. from JAX).
    """
    array_arguments: dict[str, Array] = field(init=False, default_factory=dict)
    """A mapping of unique names to arrays that have been found in the expression
    graph. These need to be added as arguments or defined as variables on code
    generation.
    """
    unique_name_generator: pytools.UniqueNameGenerator = field(
        init=False,
        default_factory=lambda: pytools.UniqueNameGenerator(forced_prefix="_arg"),
    )
    """A name generator for :attr:`array_arguments`."""

    def handle_unsupported_expression(self, expr: object, enclosing_prec: int) -> str:
        raise NotImplementedError(f"{type(self)} cannot handle {type(expr)}: {expr}")

    def map_function(self, expr: sym.Function, enclosing_prec: int) -> str:
        return f"{self.module}.{expr.name}"

    def map_numpy_array(self, expr: Array, enclosing_prec: int) -> str:
        for name, ary in self.array_arguments.items():
            if ary is expr:
                return name

        name = self.unique_name_generator("")
        self.array_arguments[name] = expr

        return name

    def map_contract(self, expr: sym.Contract, enclosing_prec: int) -> str:
        aggregate = self.rec(expr.aggregate, PREC_NONE)
        return f"{self.module}.sum({aggregate}, axis={expr.axes})"

    def map_reshape(self, expr: sym.Reshape, enclosing_prec: int) -> str:
        aggregate = self.rec(expr.aggregate, PREC_NONE)
        return f"{self.module}.reshape({aggregate}, shape={expr.shape})"

    def map_dot_product(self, expr: sym.DotProduct, enclosing_prec: int) -> str:
        left = self.rec(expr.left, PREC_NONE)  # type: ignore[arg-type]
        right = self.rec(expr.right, PREC_NONE)  # type: ignore[arg-type]
        return f"{self.module}.dot({left}, {right})"


# }}}


# {{{ target


@dataclass(frozen=True)
class NumpyTarget(Target):
    module: ClassVar[str] = "np"
    """Name of the :mod:`numpy` module. This can be changed for :mod:`numpy`
    compatible module with some work (e.g. from JAX).
    """
    funcname: ClassVar[str] = "_lambdify_generated_func_numpy"
    """The name of the generated function. This should not be seen outside of
    this code generator.
    """

    def _get_module(self) -> Any:  # noqa: PLR6301
        return np

    def _get_code_generator(self) -> NumpyCodeGenerator:
        return NumpyCodeGenerator(module=self.module)

    def generate_code(
        self,
        inputs: sym.Variable | tuple[sym.Variable, ...],
        exprs: sym.Expression | tuple[sym.Expression, ...],
        *,
        variables: sym.Variable | tuple[sym.Variable, ...] | None = None,
        sizes: int | tuple[int, ...] | None = None,
        name: str = "expr",
        pretty: bool = False,
    ) -> Code:
        if isinstance(inputs, sym.Variable):
            inputs = (inputs,)

        if isinstance(exprs, sym.Expression):
            exprs = (exprs,)

        cgen = self._get_code_generator()
        expressions = ", ".join(cgen(expr) for expr in exprs)

        from pytools.py_codegen import PythonFunctionGenerator

        args = sorted(cgen.array_arguments)
        py = PythonFunctionGenerator(
            self.funcname,
            args=(*(arg.name for arg in inputs), *args),
        )

        if variables is not None:
            # FIXME: this is a bit hacky. we basically have some variables in the
            # code that we need to define beforehand based on the input vector y.
            # This seems brittle as we add more variables to the code..
            if sizes is None:
                raise ValueError("must provide variable 'sizes'")

            if isinstance(variables, sym.Variable):
                variables = (variables,)

            if isinstance(sizes, int):
                sizes = (sizes,) * len(variables)

            i = 0
            y = inputs[-1]
            for n_i, arg in zip(sizes, variables, strict=True):
                py(f"{arg.name} = {y.name}[{i}:{i + n_i}]")
                i += n_i

        if len(exprs) == 1:
            py(f"return {cgen(exprs[0])}")
        else:
            py(f"return {self.module}.hstack([{expressions}])")

        source = py.get()
        log.debug("Code:\n%s", source)

        if pretty:
            import ast

            source = ast.unparse(ast.parse(source))

        return Code(
            name=name,
            entrypoint=self.funcname,
            source=source,
            inputs=inputs,
            args=tuple(cgen.array_arguments[k] for k in args),
            context={self.module: self._get_module()},
        )

    def lambdify(self, code: Code) -> Callable[..., Array]:  # noqa: PLR6301
        func = execute_code(code)

        def wrapper(*args: Array) -> Array:
            return func(*args, *code.args)

        return wrapper


# }}}
