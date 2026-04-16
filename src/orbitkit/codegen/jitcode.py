# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
import shutil
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

import orbitkit.symbolic.primitives as sym
from orbitkit.codegen import Assignment, Code
from orbitkit.codegen.jitcxde import (
    JiTCXDEExpression,
    JiTCXDETarget,
    cflags,
    linker_flags,
)
from orbitkit.typing import Array
from orbitkit.utils import module_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import jitcode

log = module_logger(__name__)


# {{{ target


def make_input_variable(n: int | tuple[int, ...], offset: int = 0) -> JiTCXDEExpression:
    import jitcode

    y = np.empty(n, dtype=object)
    for i, idx in enumerate(np.ndindex(y.shape)):
        y[idx] = jitcode.y(offset + i)

    return y


@dataclass(frozen=True)
class JiTCODETarget(JiTCXDETarget):
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
        debug: bool | None = None,
        # jitcode arguments
        module_location: str | pathlib.Path | None = None,
        openmp: bool = False,
        verbose: bool = False,
    ) -> jitcode.jitcode:
        import jitcode
        import symengine as sp

        if debug is None:
            debug = __debug__

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

            if module_location is not None and module_location.exists():
                from jitcxde_common.modules import find_and_load_module

                # FIXME: this is not exactly documented API
                ode.jitced = find_and_load_module(ode._modulename, ode._tmpfile())
                ode.f = ode.jitced.f
                if hasattr(ode.jitced, "jac"):
                    ode.jac = ode.jitced.jac

                ode._initialise = ode.jitced.initialise
                ode.compile_attempt = True
            else:
                t_start = time.time()
                ode.compile_C(
                    extra_compile_args=cflags(debug=debug),
                    extra_linker_args=linker_flags(debug=debug),
                    verbose=verbose,
                    omp=openmp,
                    modulename=module_location.stem if module_location else None,
                )

                if module_location is not None:
                    from jitcxde_common.modules import get_module_path

                    # FIXME: this is not exactly documented API
                    sourcefile = get_module_path(ode._modulename, ode._tmpfile())
                    shutil.copy(sourcefile, module_location)

                if verbose:
                    log.info("Compilation time: %.3fs.", time.time() - t_start)

        # NOTE: we cannot add parameters here because JiTCODE will try to compile
        # things and it won't fine the initial conditions.. it's up to the user.
        ode.set_integrator(method, atol=atol, rtol=rtol)

        return ode


# }}}
