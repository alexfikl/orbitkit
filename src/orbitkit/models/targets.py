# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from pymbolic.mapper.stringifier import PREC_NONE, StringifyMapper

import orbitkit.models.symbolic as sym
from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ numpy target


@dataclass(frozen=True)
class NumpyCodeGenerator(StringifyMapper[Any]):
    module: str = "np"

    def handle_unsupported_expression(self, expr: object, enclosing_prec: int) -> str:
        raise NotImplementedError(f"{type(self)} cannot handle {type(expr)}: {expr}")

    def map_function(self, expr: sym.Function, enclosing_prec: int) -> str:
        return f"{self.module}.{expr.name}"

    def map_numpy_array(
        self, expr: np.ndarray[tuple[int, ...], np.dtype[Any]], enclosing_prec: int
    ) -> str:
        # FIXME: this is quite hacky
        ary = repr(expr).replace("dtype=", f"dtype={self.module}.")
        return f"{self.module}.{ary}"

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

    module: str = "np"

    def generate_code(self, model: sym.Model, n: int | tuple[int, ...]) -> str:
        args, exprs = model.symbolify(n)
        gen = NumpyCodeGenerator(module=self.module)

        arguments = ", ".join(args.name for args in args)
        expressions = ", ".join(gen(expr) for expr in exprs)

        return f"lambda {arguments}: np.hstack([{expressions}])"

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
        funcname = "_numpy_lambdify_generated_func"
        filename = (
            f"<generated code for {type(model).__name__} [{NumpyTarget.codecounter}]>"
        )
        NumpyTarget.codecounter += 1

        code = self.generate_code(model, n)
        code = f"{funcname} = {code}"

        log.info("Code:\n%s", code)

        exec(
            compile(code, filename, "exec"),
            {self.module: np, "_MODULE_SOURCE_CODE": code},
            funclocals,
        )

        import linecache

        func = funclocals[funcname]
        linecache.cache[filename] = (
            len(code),
            None,
            code.splitlines(keepends=True),
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
            return func(t, *[y[s_i] for s_i in slices])  # type: ignore[no-any-return]

        return wrapper


# }}}
