# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
import shutil
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

import orbitkit.symbolic.primitives as sym
from orbitkit.codegen import Assignment, Code
from orbitkit.codegen.jitcxde import (
    JiTCXDECompiledCode,
    JiTCXDEExpression,
    JiTCXDETarget,
    cflags,
    fill_symbolic_parameters,
    has_jitcdde,
    linker_flags,
)
from orbitkit.symbolic.mappers import IdentityMapper, WalkMapper
from orbitkit.typing import Array1D
from orbitkit.utils import module_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import jitcdde
    import symengine as sp
    from jitcxde_common import jitcxde
    from pymbolic.typing import Expression as PymbolicExpression

log = module_logger(__name__)

# {{{ gather mapper


class DiracDelayReplacer(IdentityMapper):
    """A mapper that replaces all
    :class:`~orbitkit.symbolic.primitives.DiracDelayKernel` call expressions in the
    expression tree with a simple :class:`~pymbolic.primitives.Variable`.

    The resulting mapping can be obtained from :attr:`dirac_to_variable`.
    """

    dirac_to_variable: dict[sym.Call, sym.Variable]
    """A mapping of replaced :class:`~orbitkit.symbolic.primitives.DiracDelayKernel`
    call expressions. Note that this class reserves the ``_ok_dde_delay_`` prefix
    for its variable names.
    """

    def __init__(self, inputs: tuple[sym.Variable, ...]) -> None:
        from pytools import UniqueNameGenerator

        self.unique_name_generator = UniqueNameGenerator(forced_prefix="_ok_dde_delay_")
        self.dirac_to_variable = {}
        self.name_to_inputs = {inp.name: inp for inp in inputs}

    def map_call(self, expr: sym.Call) -> PymbolicExpression:
        func = expr.function
        if not isinstance(func, sym.DelayKernel):
            return super().map_call(expr)

        if not isinstance(func, sym.DiracDelayKernel):
            raise ValueError(f"found non-Dirac kernel: {expr}")

        (y,) = expr.parameters
        if not isinstance(y, sym.Variable):
            raise NotImplementedError(
                f"cannot delay non-Variable expression: {y} (type {type(y)})"
            )

        if y.name not in self.name_to_inputs:
            raise ValueError(f"variable '{y}' is not a known input")

        # FIXME: what do we want to allow here? jitcdde supports pretty arbitrary
        # expressions, including state-dependent, so it doesn't make sense to be
        # too restrictive? Maybe just let it fail later..
        # if not isinstance(func.tau, (int, float, sym.Variable)):
        #     raise NotImplementedError(f"delay 'tau' must be a number: {func.tau}")

        inp = self.name_to_inputs[y.name]
        try:
            return self.dirac_to_variable[expr]
        except KeyError:
            self.dirac_to_variable[expr] = result = replace(
                inp, name=self.unique_name_generator(f"{y.name}")
            )

            return result


class DiracDelayGatherer(WalkMapper):
    def __init__(self) -> None:
        self.delays: set[sym.Expression] = set()

    def visit(self, expr: object) -> bool:
        if isinstance(expr, sym.DiracDelayKernel):
            self.delays.add(expr.tau)
            return False

        return True


def find_discrete_delays(
    expr: sym.Expression | tuple[sym.Expression, ...],
) -> tuple[sym.Expression, ...]:
    gather = DiracDelayGatherer()
    gather(expr)

    return tuple(gather.delays)


# }}}


# {{{ CompiledCode


@dataclass(frozen=True)
class JiTCDDECompiledCode(JiTCXDECompiledCode):
    dde: jitcdde.jitcdde
    """The ``jitcdde`` integrator set up for this module."""

    delays: tuple[sym.Expression, ...]
    """A list of (possibly symbolic) delays that have been found in this module.
    The delays are evaluated and update in
    :meth:`~orbitkit.codegen.jitcxde.JiTCXDECompiledCode.set_parameters` once all
    the parameters are known.
    """

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["max_delay"] = float(self.dde.max_delay or 1.0)

        # FIXME: these are the parameters that we set in compile, but jitcdde
        # has a lot more that the user can just set themselves.. oh well.
        state["atol"] = float(getattr(self.dde, "atol", 1.0e-10))
        state["rtol"] = float(getattr(self.dde, "rtol", 1.0e-5))
        state["first_step"] = float(getattr(self.dde, "dt", 1.0))
        state["max_step"] = float(getattr(self.dde, "max_step", 1.0))
        del state["dde"]

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        import symengine as sp

        module_location = state["module_location"]
        control_pars = tuple(sp.Symbol(p, real=True) for p in state["parameters"])

        dde = initialize_jitcdde(
            state["f"],
            state["y"],
            nlyapunov=state["nlyapunov"],
            control_pars=control_pars,
            module_location=module_location,
            max_delay=state.pop("max_delay"),
        )
        reload_jitcdde(dde, module_location)
        dde.set_integration_parameters(
            atol=state.pop("atol"),
            rtol=state.pop("rtol"),
            first_step=state.pop("first_step"),
            max_step=state.pop("max_step"),
        )

        state["dde"] = dde
        object.__setattr__(self, "__dict__", state)

    def reset(self) -> None:
        self.dde.purge_past()
        self.dde.delays = None
        self.dde.max_delay = None
        self.dde.integration_parameters_set = False

    def set_initial_conditions(
        self,
        y: Array1D[np.floating[Any]],
        t: float = 0.0,
    ) -> None:
        # NOTE: initialize the Lyapunov exponent equations as well. Not sure
        # this actually does anything useful.. might have just been a bug.
        if self.nlyapunov > 0:
            ylyap = np.eye(y.size, self.nlyapunov, dtype=y.dtype).T.reshape(-1)
            y = np.concatenate([y, ylyap])

        self.dde.constant_past(y, time=t)

    def set_parameters(self, **kwargs: Any) -> None:
        if not self.parameters:
            return

        # NOTE: these need to be in the same order as in JiTCODETarget.compile
        control_pars = []
        for param in self.parameters:
            if param not in kwargs:
                raise ValueError(f"parameter missing: {param}")
            control_pars.append(kwargs[param])

        from pymbolic.mapper.evaluator import evaluate

        delays = tuple(evaluate(delay, context=kwargs) for delay in self.delays)
        if 0.0 not in delays:
            delays = (0.0, *delays)

        self.dde.delays = delays
        self.dde.max_delay = max(delays)

        self.dde.set_parameters(tuple(control_pars))

    def integrate(
        self, t: float | np.floating[Any]
    ) -> tuple[
        Array1D[np.floating[Any]],
        Array1D[np.floating[Any]] | None,
        Array1D[np.floating[Any]] | None,
    ]:
        if self.nlyapunov:
            return self.dde.integrate(t)
        else:
            return self.dde.integrate(t), None, None

    def adjust_diff(self) -> None:
        self.dde.adjust_diff()

    def step_on_discontinuities(self) -> None:
        self.dde.step_on_discontinuities()


# }}}


# {{{ target


def make_input_variable(
    n: int | tuple[int, ...],
    tau: int | float | sp.Symbol | Array1D[Any] = 0,
    offset: int = 0,
) -> JiTCXDEExpression:
    import jitcdde
    import symengine as sp

    if isinstance(tau, (int, float, sp.Expr)):
        tau = np.full(n, tau)

    y = np.empty(n, dtype=object)
    for i, idx in enumerate(np.ndindex(y.shape)):
        y[idx] = jitcdde.y(offset + i, jitcdde.t - tau[idx])

    return y


def make_delay_variable(
    ys: JiTCXDEExpression,
    tau: int | float | sp.Symbol | Array1D[Any] = 0,
) -> JiTCXDEExpression:
    import jitcdde
    import symengine as sp

    if isinstance(tau, (int, float, sp.Expr)):
        tau = np.full(ys.shape, tau)

    if tau.shape != ys.shape:
        raise ValueError(
            f"'tau' shape does not match inputs: {tau.shape} (expected {ys.shape})"
        )

    result = np.empty_like(ys)
    for idx in np.ndindex(ys.shape):
        y = ys[idx]
        assert isinstance(y, sp.Function), type(y)
        assert len(y.args) == 1, y.size

        (i,) = y.args
        assert isinstance(i, sp.Integer), type(i)

        result[idx] = jitcdde.y(i, jitcdde.t - tau[idx])

    return result


def initialize_jitcdde(
    f: Array1D[Any],
    y: Array1D[Any],
    *,
    nlyapunov: int = 0,
    control_pars: Sequence[sp.Symbol],
    module_location: pathlib.Path | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> jitcdde.jitcdde:
    import jitcdde

    if "max_delay" not in kwargs:
        raise ValueError("'max_delay' not provided")
    max_delay = float(kwargs["max_delay"])

    if nlyapunov > 0:
        return jitcdde.jitcdde_lyap(
            f,
            n=y.size,
            n_lyap=nlyapunov,
            verbose=verbose,
            max_delay=max_delay,
            control_pars=control_pars,
            module_location=str(module_location) if module_location else None,
        )
    else:
        return jitcdde.jitcdde(
            f,
            n=y.size,
            verbose=verbose,
            max_delay=max_delay,
            control_pars=control_pars,
            module_location=str(module_location) if module_location else None,
        )


def compile_jitcdde(
    dde: jitcdde.jitcdde,
    *,
    module_location: pathlib.Path | None = None,
    simplify: bool = False,
    debug: bool = False,
    openmp: bool = False,
    verbose: bool = False,
) -> None:
    dde.compile_C(
        simplify=simplify,
        do_cse=False,
        # FIXME: jitcdde assumes lists
        extra_compile_args=list(cflags(debug=debug)),
        extra_link_args=list(linker_flags(debug=debug)),
        verbose=verbose,
        chunk_size=32,
        omp=openmp,
        modulename=module_location.stem if module_location else None,
    )

    if module_location is not None:
        from jitcxde_common.modules import get_module_path

        # FIXME: this is not exactly documented API
        # FIXME: does this work with what reload_module is doing?
        sourcefile = get_module_path(dde._modulename, dde._tmpfile())
        shutil.copy(sourcefile, module_location)


def reload_jitcdde(
    dde: jitcdde.jitcdde,
    module_location: pathlib.Path | None = None,
) -> None:
    if module_location is None:
        return

    if not module_location.exists():
        log.warning("Module location does not exist: %s.", module_location)
        return

    from jitcxde_common.modules import find_and_load_module

    # FIXME: this is not exactly documented API
    try:
        dde.jitced = find_and_load_module(
            module_location.stem, str(module_location.parent)
        )
    except Exception as exc:
        log.error("Failed to reload module: %s.", exc, exc_info=exc)

    dde._modulename = module_location.stem
    dde.compile_attempt = True


@dataclass(frozen=True)
class JiTCDDETarget(JiTCXDETarget):
    nlyapunov: int = 0
    """Number of Lyapunov exponents to calculate. Setting this to the default 0
    does not compute Lyapunov exponents at all.
    """

    if __debug__:

        def __post_init__(self) -> None:
            if self.nlyapunov < 0:
                raise ValueError(
                    f"invalid number of Lyapunov exponents: {self.nlyapunov}"
                )

    @staticmethod
    def has_jitcdde() -> bool:
        return has_jitcdde()

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
        delays = find_discrete_delays(exprs)
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
        code = super().generate_code(
            inputs,
            exprs,
            assignments=(*assignments, *delay_assignments),
            name=name,
            pretty=pretty,
        )

        from constantdict import constantdict

        return replace(
            code,
            context=constantdict({
                **code.context,
                "delays": delays,
                make_delay_func.name: make_delay_variable,
            }),
        )

    def initialize_module(
        self,
        f: Array1D[Any],
        y: Array1D[Any],
        *,
        control_pars: Sequence[sp.Symbol],
        module_location: pathlib.Path | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> jitcdde.jitcdde:
        return initialize_jitcdde(
            f,
            y,
            nlyapunov=self.nlyapunov,
            control_pars=control_pars,
            module_location=module_location,
            verbose=verbose,
            **kwargs,
        )

    def compile_module(  # noqa: PLR6301
        self,
        de: jitcxde,
        *,
        module_location: pathlib.Path | None = None,
        simplify: bool = False,
        debug: bool = False,
        openmp: bool = False,
        verbose: bool = False,
    ) -> None:
        import time

        import jitcdde

        t_start = time.time()

        assert isinstance(de, jitcdde.jitcdde)
        compile_jitcdde(
            de,
            module_location=module_location,
            simplify=simplify,
            debug=debug,
            openmp=openmp,
            verbose=verbose,
        )

        if verbose:
            log.info("Compilation time: %.3fs.", time.time() - t_start)

    def reload_module(  # noqa: PLR6301
        self,
        de: jitcxde,
        module_location: pathlib.Path | None = None,
    ) -> None:
        import jitcdde

        assert isinstance(de, jitcdde.jitcdde)
        return reload_jitcdde(de, module_location)

    def compile(
        self,
        code: Code,
        *,
        max_delay: float | None = None,
        parameters: Mapping[str, Any] | None = None,
        # jitcdde arguments
        atol: float = 1.0e-10,
        rtol: float = 1.0e-5,
        first_step: float | None = None,
        max_step: float = 1.0,
        module_location: str | pathlib.Path | None = None,
        simplify: bool = False,
        openmp: bool = False,
        debug: bool | None = None,
        verbose: bool = False,
    ) -> JiTCDDECompiledCode:
        import jitcdde
        import symengine as sp

        if debug is None:
            debug = __debug__

        if isinstance(module_location, str):
            module_location = pathlib.Path(module_location)

        # FIXME: this assume that we have (t, y) as our inputs always. This
        # should be made less implicit, so we don't have to guess.
        assert len(code.inputs) == 2
        _, inputs = code.inputs
        assert isinstance(inputs, sym.MatrixSymbol)

        # generate Python code
        parameters = fill_symbolic_parameters(code, parameters)
        func = self.lambdify(code, parameters=parameters)

        # evaluate to obtain symengine expressions
        y = make_input_variable(inputs.size)
        f = func(jitcdde.t, y)

        # get control parameters, if any
        control_pars = tuple(
            param for param in parameters.values() if isinstance(param, sp.Symbol)
        )

        # determine max_delay
        # NOTE: max_delay is not actually used in code generation, but is
        # required by jitcdde for some internal machinery. We will reset it later
        # when all the parameters are known, so it's fine if this isn't right
        delays = code.context.get("delays", [])
        if max_delay is None:
            max_delay = max(
                (tau for tau in delays if isinstance(tau, (int, float))),
                default=1.0,
            )

        # compile
        # NOTE: reload logic:
        # 1. If given a module_location that exists
        #    => let jitcxde reload it
        # 2. If given a module location that does not exist
        #    => compile it
        #    => load it
        if module_location and module_location.exists():
            dde = self.initialize_module(
                f,
                y,
                verbose=verbose,
                max_delay=max_delay,
                control_pars=control_pars,
                module_location=module_location,
            )
        else:
            dde = self.initialize_module(
                f,
                y,
                max_delay=max_delay,
                control_pars=control_pars,
            )
            self.compile_module(
                dde,
                module_location=module_location,
                simplify=simplify,
                debug=debug,
                openmp=openmp,
                verbose=verbose,
            )

            self.reload_module(dde, module_location)

        # NOTE: first_step is 1 by default: if the delays are < 1.0, then it
        # will needlessly start from a too large step. We try to help it out..
        if first_step is None:
            first_step = max_delay / 2 if max_delay > 0 else max_step

        dde.set_integration_parameters(
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_step=max(max_step, first_step),
        )

        return JiTCDDECompiledCode(
            f=f,
            y=y,
            parameters=code.parameters,
            module_location=module_location,
            nlyapunov=self.nlyapunov,
            dde=dde,
            delays=delays,
        )


# }}}
