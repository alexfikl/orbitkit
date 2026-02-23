# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

import numpy as np

import orbitkit.symbolic.primitives as sym
from orbitkit.codegen import Assignment, Code
from orbitkit.codegen.numpy import NumpyCodeGenerator, NumpyTarget
from orbitkit.typing import Array
from orbitkit.utils import module_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import jitcode

log = module_logger(__name__)

JITCODE_COMMON_CFLAGS = [
    "-std=c11",
    "-march=native",
    "-mtune=native",
    "-Wno-unknown-pragmas",
]

JITCODE_RELEASE_CFLAGS = [
    *JITCODE_COMMON_CFLAGS,
    "-O3",
    "-ffast-math",
    "-g0",
]
"""Compiler flags used for release builds of the JiTCODE module."""

JITCODE_DEBUG_CFLAGS = [
    *JITCODE_COMMON_CFLAGS,
    "-O0",
    "-ggdb",
]
"""Compiler flags used for debug builds of the JiTCODE module."""

# {{{ target

JiTCODEExpression: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[Any]]


def make_input_variable(n: int | tuple[int, ...], offset: int = 0) -> JiTCODEExpression:
    import jitcode

    y = np.empty(n, dtype=object)
    for i, idx in enumerate(np.ndindex(y.shape)):
        y[idx] = jitcode.y(offset + i)

    return y


@dataclass(frozen=True)
class JiTCODECodeGenerator(NumpyCodeGenerator):
    sym_module: str = "sp"

    def map_function(self, expr: sym.Function, enclosing_prec: int) -> str:
        return f"vectorized({self.sym_module}.{expr.name})"


@dataclass(frozen=True)
class JiTCODETarget(NumpyTarget):
    module: ClassVar[str] = "np"
    sym_module: ClassVar[str] = "sp"
    funcname: ClassVar[str] = "_lambdify_generated_func_jitcode_symengine"

    def _get_code_generator(self, inputs: set[str]) -> NumpyCodeGenerator:
        return JiTCODECodeGenerator(
            inputs=inputs, module=self.module, sym_module=self.sym_module
        )

    def generate_code(
        self,
        inputs: sym.Variable | tuple[sym.Variable, ...],
        exprs: sym.Expression | tuple[sym.Expression, ...],
        *,
        assignments: tuple[Assignment, ...] | None = None,
        name: str = "expr",
        pretty: bool = False,
    ) -> Code:
        if assignments is None:
            raise NotImplementedError("JiTCODE cannot generate individual functions")

        import symengine
        from pytools.obj_array import vectorized

        code = super().generate_code(
            inputs,
            exprs,
            assignments=assignments,
            name=name,
            pretty=pretty,
        )
        log.debug("Code:\n%s", code.source)

        return replace(
            code,
            context={
                **code.context,
                self.sym_module: symengine,
                "vectorized": vectorized,
            },
        )

    def lambdify(
        self,
        code: Code,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> Callable[..., Array]:
        # NOTE: if we need extra parameters, just add them as symbols. These
        # will be added properly according to JiTCODE in the compile function.
        if code.parameters:
            import symengine as sp

            if parameters is None:
                parameters = {}

            for param in code.parameters:
                if param not in parameters:
                    parameters[param] = sp.Symbol(param)

        return super().lambdify(code, parameters=parameters)

    def compile(  # noqa: PLR6301
        self,
        f: Array,
        y: Array,
        *,
        method: str = "RK45",
        atol: float = 1.0e-6,
        rtol: float = 1.0e-8,
        parameters: tuple[str, ...] = (),
        debug: bool = False,
        # jitcode arguments
        module_location: str | pathlib.Path | None = None,
        simplify: bool = False,
        openmp: bool = False,
        verbose: bool = False,
    ) -> jitcode.jitcode:
        import jitcode
        import symengine as sp

        if module_location is not None:
            module_location = pathlib.Path(module_location)

        control_pars = tuple(sp.Symbol(param) for param in parameters)

        if module_location and module_location.exists():
            ode = jitcode.jitcode(
                f,
                n=y.size,
                control_pars=control_pars,
                verbose=verbose,
                module_location=str(module_location),
            )
        else:
            ode = jitcode.jitcode(
                f,
                n=y.size,
                control_pars=control_pars,
                verbose=verbose,
            )

            ode.compile_C(
                simplify=simplify,
                do_cse=False,
                extra_compile_args=(
                    JITCODE_DEBUG_CFLAGS if debug else JITCODE_RELEASE_CFLAGS
                ),
                verbose=verbose,
                chunk_size=32,
                modulename=str(module_location) if module_location else None,
                omp=openmp,
            )

            if module_location is not None:
                t_start = time.time()
                newfilename = ode.save_compiled(str(module_location), overwrite=True)
                if verbose:
                    log.info("Compilation time: %.3fs.", time.time() - t_start)

                if newfilename != str(module_location):
                    log.warning(
                        "jitcode saved compiled module in different file: '%s'. "
                        "This may cause performance issues since it will be recompiled",
                        newfilename,
                    )

        # NOTE: we cannot add parameters here because JiTCODE will try to compile
        # things and it won't fine the initial conditions.. it's up to the user.
        ode.set_integrator(method, atol=atol, rtol=rtol)

        return ode


# }}}
