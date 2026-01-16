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
    import jitcode

log = module_logger(__name__)


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

    def _get_code_generator(self) -> NumpyCodeGenerator:
        return JiTCODECodeGenerator(module=self.module, sym_module=self.sym_module)

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

    def compile(  # noqa: PLR6301
        self,
        f: Array,
        y: Array,
        *,
        method: str = "RK45",
        atol: float = 1.0e-6,
        rtol: float = 1.0e-8,
        module_location: str | pathlib.Path | None = None,
        verbose: bool = False,
    ) -> jitcode.jitcode:
        import jitcode

        if module_location is not None:
            module_location = pathlib.Path(module_location)

        if module_location and module_location.exists():
            ode = jitcode.jitcode(
                f,
                n=y.size,
                verbose=verbose,
                module_location=str(module_location),
            )
        else:
            ode = jitcode.jitcode(
                f,
                n=y.size,
                verbose=verbose,
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

        ode.set_integrator(method, atol=atol, rtol=rtol)
        return ode


# }}}
