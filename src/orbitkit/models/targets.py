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
    name: str
    """An identifier for this chunk of code."""
    source: str
    """Source code obtained from a symbolic expression."""
    inputs: tuple[sym.Variable, ...]
    """Inputs for the *source* expression."""
    args: tuple[Array, ...]
    """Additional arguments required for the *source* expression."""

    def __str__(self) -> str:
        return self.source


# }}}

# {{{ numpy target


# TODO: will need to make this into a proper compiler with statements and
# assignments and whatnot at some point. The main driving force for that is
# the need for CSE to save intermediate results..


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
    unique_name_generator: UniqueNameGenerator = field(
        init=False,
        default_factory=lambda: UniqueNameGenerator(forced_prefix="_arg"),
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


@dataclass(frozen=True)
class NumpyTarget:
    module: str = "np"
    """Name of the :mod:`numpy` module. This can be changed for :mod:`numpy`
    compatible module with some work (e.g. from JAX).
    """
    funcname: str = "_numpy_lambdify_generated_func"
    """The name of the generated function. This should not be seen outside of
    this code generator.
    """

    codecounter: ClassVar[int] = 0
    """A counter to ensure uniqueness of some generated functions."""

    def generate_model_code(
        self,
        model: sym.Model,
        n: int | tuple[int, ...],
        *,
        pretty: bool = False,
    ) -> Code:
        """Generate code for a given *model*.

        :arg n: expected size of the input variables. These can all be different
            if te model takes multiple variables.
        :arg pretty: if *True*, some simple code formatting is performed.
        """

        if isinstance(n, int):
            n = (n,) * len(model.variables)

        inputs, exprs = model.symbolify(n)
        to_numpy = NumpyCodeGenerator(module=self.module)
        expressions = ", ".join(to_numpy(expr) for expr in exprs)

        from pytools.py_codegen import PythonFunctionGenerator

        y = sym.MatrixSymbol("y", (sum(n),))
        args = sorted(to_numpy.array_arguments)
        cgen = PythonFunctionGenerator(
            self.funcname,
            args=(inputs[0].name, y.name, *args),
        )

        i = 0
        for n_i, arg in zip(n, inputs[1:], strict=True):
            cgen(f"{arg.name} = {y.name}[{i}:{i + n_i}]")
            i += n_i

        cgen(f"return np.hstack([{expressions}])")

        source = cgen.get()
        log.debug("Code:\n%s", source)

        if pretty:
            import ast

            # NOTE: this just adds some nicer spaces around operators, so not
            # really the code prettifier one would hope for
            source = ast.unparse(ast.parse(source))

        return Code(
            name=type(model).__name__,
            source=source,
            inputs=(inputs[0], y),
            args=tuple(to_numpy.array_arguments[k] for k in args),
        )

    def generate_code(
        self,
        inputs: tuple[sym.Variable],
        expr: sym.Expression,
        *,
        name: str = "expr",
        pretty: bool = False,
    ) -> Code:
        """Generate code for an arbitrary expression *expr*.

        :arg inputs: input variables required to evaluate the expression *expr*.
        :arg name: an identifier for the generated function.
        :arg pretty: if *True*, some simple code formatting is performed.
        """

        from pytools.py_codegen import PythonFunctionGenerator

        to_numpy = NumpyCodeGenerator(module=self.module)
        cgen = PythonFunctionGenerator(
            self.funcname,
            args=tuple(arg.name for arg in inputs),
        )
        cgen(f"return {to_numpy(expr)}")

        source = cgen.get()
        log.debug("Code:\n%s", source)

        if pretty:
            import ast

            source = ast.unparse(ast.parse(source))

        return Code(name=name, source=source, inputs=inputs, args=())

    def lambdify_model(
        self, model: sym.Model, n: int | tuple[int, ...]
    ) -> Callable[[float, Array], Array]:
        """Create a callable function for the given model."""
        code = self.generate_model_code(model, n)
        return self.lambdify(code)

    def lambdify(self, code: Code) -> Callable[..., Array]:
        """Create a callable function for some arbitrary code.

        Note that the code is assumed to be generated by this target. E.g., we
        expect there to exist a function with the name :attr:`funcname` that
        can be imported.

        .. warning::

            Like :func:`sympy.utilities.lambdify.lambdify`, this function uses
            :func:`exec` to generate a callable. This is not very safe for
            arbitrary code, so use with care!
        """

        funclocals: dict[str, Any] = {}
        filename = f"<generated code for {code.name} [{NumpyTarget.codecounter}]>"
        NumpyTarget.codecounter += 1

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

        def wrapper(*args: Array) -> Array:
            return func(*args, *code.args)  # type: ignore[no-any-return]

        return wrapper


# }}}
