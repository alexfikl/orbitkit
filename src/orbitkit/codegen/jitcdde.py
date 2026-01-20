# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

import numpy as np
import pymbolic.primitives as prim

import orbitkit.symbolic.primitives as sym
from orbitkit.codegen import Assignment, Code
from orbitkit.codegen.jitcode import JiTCODECodeGenerator, JiTCODETarget
from orbitkit.symbolic.mappers import IdentityMapper
from orbitkit.typing import Array
from orbitkit.utils import module_logger

if TYPE_CHECKING:
    import jitcdde
    from pymbolic.typing import Expression as PymbolicExpression

    from orbitkit.codegen.numpy import NumpyCodeGenerator

log = module_logger(__name__)


# {{{ gather mapper


class DiracDelayReplacer(IdentityMapper):
    """A mapper that replaces all
    :class:`~orbitkit.symbolic.primitives.DiracDelayKernel` call expressions in the
    expression tree with a simple :class:`~pymbolic.primitives.Variable`.

    The resulting mapping can be obtained from :attr:`dirac_to_variable`.
    """

    dirac_to_variable: dict[prim.Call, sym.Variable]
    """A mapping of replaced :class:`~orbitkit.symbolic.primitives.DiracDelayKernel`
    call expressions. Note that this class reserves the ``_ok_dde_delay_`` prefix
    for its variable names.
    """

    def __init__(self, inputs: tuple[sym.Variable, ...]) -> None:
        from pytools import UniqueNameGenerator

        self.unique_name_generator = UniqueNameGenerator(forced_prefix="_ok_dde_delay_")
        self.dirac_to_variable = {}
        self.name_to_inputs = {inp.name: inp for inp in inputs}

    def map_call(self, expr: prim.Call) -> PymbolicExpression:
        func = expr.function
        if not isinstance(func, sym.DelayKernel):
            if isinstance(func, sym.DiracDelayKernel):
                return super().map_call(expr)
            else:
                raise ValueError(f"found non-Dirac kernel: {expr}")

        assert isinstance(func, sym.DiracDelayKernel)

        (y,) = expr.parameters
        if not isinstance(y, sym.Variable):
            raise NotImplementedError(
                f"cannot delay non-Variable expression: {y} (type {type(y)})"
            )

        if y.name not in self.name_to_inputs:
            raise ValueError(f"variable '{y}' is not a known input")

        if not isinstance(func.tau, (int, float)):
            raise NotImplementedError(f"delay 'tau' must be a number: {func.tau}")

        inp = self.name_to_inputs[y.name]
        try:
            return self.dirac_to_variable[expr]
        except KeyError:
            self.dirac_to_variable[expr] = result = replace(
                inp, name=self.unique_name_generator(f"{y.name}")
            )

            return result


# }}}


# {{{ target


JiTCDDEExpression: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[Any]]


def make_input_variable(
    n: int | tuple[int, ...], tau: int | float = 0, offset: int = 0
) -> JiTCDDEExpression:
    import jitcdde

    y = np.empty(n, dtype=object)
    for i, idx in enumerate(np.ndindex(y.shape)):
        y[idx] = jitcdde.y(offset + i, jitcdde.t - tau)

    return y


def make_delay_variable(
    ys: JiTCDDEExpression, tau: int | float = 0
) -> JiTCDDEExpression:
    import jitcdde
    import symengine as sp

    result = np.empty_like(ys)
    for idx in np.ndindex(ys.shape):
        y = ys[idx]
        assert isinstance(y, sp.Function), type(y)
        assert len(y.args) == 1, y.size

        (i,) = y.args
        assert isinstance(i, sp.Integer), type(i)

        result[idx] = jitcdde.y(i, jitcdde.t - tau)

    return result


@dataclass(frozen=True)
class JiTCDDECodeGenerator(JiTCODECodeGenerator):
    pass


class JiTCDDETarget(JiTCODETarget):
    module: ClassVar[str] = "np"
    sym_module: ClassVar[str] = "sp"
    funcname: ClassVar[str] = "_lambdify_generated_func_jitcdde_symengine"

    def _get_code_generator(self) -> NumpyCodeGenerator:
        return JiTCDDECodeGenerator(module=self.module, sym_module=self.sym_module)

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
            raise NotImplementedError("JiTCDDE cannot generate individual functions")

        if isinstance(inputs, sym.Variable):
            inputs = (inputs,)

        # gather all delayed variables
        mapper = DiracDelayReplacer((
            *inputs,
            *(assign.assignee for assign in assignments),
        ))
        exprs = mapper(exprs)  # ty: ignore[invalid-assignment]
        if not mapper.dirac_to_variable:
            raise ValueError(
                "code does not contain any delayed variables (use JiTCODETarget)"
            )

        # create delayed variables using jitcdde
        from pymbolic.primitives import Call

        make_delay_func = sym.Variable("make_delay_variable")

        delay_assignments = []
        for expr, var in mapper.dirac_to_variable.items():
            assert isinstance(var, sym.MatrixSymbol)

            kernel = expr.function
            assert isinstance(kernel, sym.DiracDelayKernel)
            (y,) = expr.parameters
            assert isinstance(y, sym.Variable)

            delay_assignments.append(
                Assignment(var, Call(make_delay_func, (y, kernel.tau)))
            )

        # generate code
        import symengine
        from pytools.obj_array import vectorized

        code = super().generate_code(
            inputs,
            exprs,
            assignments=(*assignments, *delay_assignments),
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
                make_delay_func.name: make_delay_variable,
            },
        )

    def compile(  # ty: ignore[invalid-method-override] # noqa: PLR6301
        self,
        f: Array,
        y: Array,
        *,
        max_delay: float,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-8,
        module_location: str | pathlib.Path | None = None,
        verbose: bool = False,
    ) -> jitcdde.jitcdde:
        import jitcdde

        if module_location is not None:
            module_location = pathlib.Path(module_location)

        if module_location and module_location.exists():
            dde = jitcdde.jitcdde(
                f,
                n=y.size,
                verbose=verbose,
                max_delay=max_delay,
                module_location=str(module_location),
            )
        else:
            dde = jitcdde.jitcdde(
                f,
                n=y.size,
                verbose=verbose,
            )

            if module_location is not None:
                t_start = time.time()
                newfilename = dde.save_compiled(str(module_location), overwrite=True)
                if verbose:
                    log.info("Compilation time: %.3fs.", time.time() - t_start)

                if newfilename != str(module_location):
                    log.warning(
                        "jitcdde saved compiled module in different file: '%s'. "
                        "This may cause performance issues since it will be recompiled",
                        newfilename,
                    )

        dde.set_integration_parameters(rtol=rtol, atol=atol)
        return dde


# }}}
