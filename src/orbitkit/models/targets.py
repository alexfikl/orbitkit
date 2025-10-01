# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from pymbolic.mapper.stringifier import PREC_NONE, StringifyMapper
from pytools import UniqueNameGenerator

import orbitkit.models.symbolic as sym
from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ code


@dataclass
class Code:
    source: str
    inputs: tuple[sym.Variable, ...]
    args: dict[str, Array]

    def __str__(self) -> str:
        return self.source


# }}}

# {{{ numpy target


# TODO: will need to make this into a proper compiler with statements and
# assignments and whatnot at some point. The main driving force for that is
# the need for CSE to save intermediate results..


@dataclass(frozen=True)
class NumpyCodeGenerator(StringifyMapper[Any]):
    module: str = "np"
    array_arguments: dict[str, Array] = field(init=False, default_factory=dict)
    unique_names: UniqueNameGenerator = field(
        init=False,
        default_factory=lambda: UniqueNameGenerator(forced_prefix="_arg"),
    )

    def handle_unsupported_expression(self, expr: object, enclosing_prec: int) -> str:
        raise NotImplementedError(f"{type(self)} cannot handle {type(expr)}: {expr}")

    def map_function(self, expr: sym.Function, enclosing_prec: int) -> str:
        return f"{self.module}.{expr.name}"

    def map_numpy_array(self, expr: Array, enclosing_prec: int) -> str:
        for name, ary in self.array_arguments.items():
            if ary is expr:
                return name

        name = self.unique_names("")
        self.array_arguments[name] = expr

        return name

    def map_contract(self, expr: sym.Contract, enclosing_prec: int) -> str:
        aggregate = self.rec(expr.aggregate, PREC_NONE)
        return f"{self.module}.sum({aggregate}, axis={expr.axis})"

    def map_reshape(self, expr: sym.Reshape, enclosing_prec: int) -> str:
        aggregate = self.rec(expr.aggregate, PREC_NONE)
        return f"{self.module}.reshape({aggregate}, shape={expr.shape})"

    def map_dot_product(self, expr: sym.DotProduct, enclosing_prec: int) -> str:
        left = self.rec(expr.left, PREC_NONE)  # type: ignore[arg-type]
        right = self.rec(expr.right, PREC_NONE)  # type: ignore[arg-type]
        return f"{self.module}.dot({left}, {right})"


@dataclass(frozen=True)
class NumpyTarget:
    codecounter: ClassVar[int] = 0
    funcname: str = "_numpy_lambdify_generated_func"

    module: str = "np"

    def generate_code(
        self,
        model: sym.Model,
        n: int | tuple[int, ...],
        *,
        pretty: bool = False,
    ) -> Code:
        inputs, exprs = model.symbolify(n)
        gen = NumpyCodeGenerator(module=self.module)
        expressions = ", ".join(gen(expr) for expr in exprs)

        from pytools.py_codegen import PythonFunctionGenerator

        cgen = PythonFunctionGenerator(
            self.funcname,
            args=(*(arg.name for arg in inputs), *gen.array_arguments),
        )
        cgen(f"return np.hstack([{expressions}])")

        source = cgen.get()
        if pretty:
            import ast

            # NOTE: this just adds some nicer spaces around operators, so not
            # really the code prettifier one would hope for
            source = ast.unparse(ast.parse(source))

        return Code(source=source, inputs=inputs, args=gen.array_arguments.copy())

    def lambdify(
        self,
        model: sym.Model,
        n: int | tuple[int, ...],
    ) -> Callable[[float, Array], Array]:
        """Create a callable that is usable by :func:`scipy.integrate.solve_ivp`
        or other similar integrators.

        This uses :meth:`variables` and :func:`lambdify` to create a
        :mod:`numpy` compatible callable.
        """
        if isinstance(n, int):
            n = (n,) * len(model.variables)

        funclocals: dict[str, Any] = {}
        filename = (
            f"<generated code for {type(model).__name__} [{NumpyTarget.codecounter}]>"
        )
        NumpyTarget.codecounter += 1

        code = self.generate_code(model, n)
        log.info("Code:\n%s", code)

        exec(
            compile(code.source, filename, "exec"),
            {self.module: np, "_MODULE_SOURCE_CODE": code.source},
            funclocals,
        )

        import linecache

        func = funclocals[self.funcname]
        linecache.cache[filename] = (
            len(code.source),
            None,
            code.source.splitlines(keepends=True),
            filename,
        )

        import weakref

        def make_finalize(filename: str) -> Callable[[], None]:
            def _cleanup() -> None:
                if filename in linecache.cache:
                    del linecache.cache[filename]

            return _cleanup

        weakref.finalize(func, make_finalize(filename))

        i = 0
        slices = []
        for n_i in n:
            slices.append(np.s_[i : i + n_i])
            i += n_i

        def wrapper(t: float, y: Array) -> Array:
            return func(t, *[y[s_i] for s_i in slices], **code.args)  # type: ignore[no-any-return]

        return wrapper


# }}}
