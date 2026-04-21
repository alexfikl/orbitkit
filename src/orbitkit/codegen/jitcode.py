# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

import orbitkit.symbolic.primitives as sym
from orbitkit.codegen import Code
from orbitkit.codegen.jitcxde import (
    JiTCXDECompiledCode,
    JiTCXDEExpression,
    JiTCXDETarget,
    cflags,
    fill_symbolic_parameters,
    has_jitcode,
    linker_flags,
)
from orbitkit.typing import Array1D
from orbitkit.utils import module_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import jitcode
    import symengine as sp
    from jitcxde_common import jitcxde

log = module_logger(__name__)


# {{{ compiled code


@dataclass(frozen=True)
class JiTCODECompiledCode(JiTCXDECompiledCode):
    ode: jitcode.jitcode
    """An instance of the underlying ``jitcode`` integrator."""

    integrator_name: str
    """Underlying integrator used by ``jitcode``."""
    integrator_params: Mapping[str, Any]
    """Additional parameters use for the ``jitcode`` integrator."""

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        del state["ode"]

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        import symengine as sp

        module_location = state["module_location"]
        control_pars = tuple(sp.Symbol(p, real=True) for p in state["parameters"])

        ode = initialize_jitcode(
            state["f"],
            state["y"],
            nlyapunov=state["nlyapunov"],
            control_pars=control_pars,
            module_location=module_location,
        )
        reload_jitcode(ode, module_location)
        ode.set_integrator(
            state["integrator_name"],
            **state["integrator_params"],
        )

        state["ode"] = ode
        object.__setattr__(self, "__dict__", state)

    def reset(self) -> None:
        pass

    def set_initial_conditions(
        self,
        y: Array1D[np.floating[Any]],
        t: float = 0.0,
    ) -> None:
        self.ode.set_initial_value(y, time=t)

    def set_parameters(self, **kwargs: Any) -> None:
        if len(kwargs) == 0:
            return

        control_pars = []
        for param in self.parameters:
            if param not in kwargs:
                raise ValueError(f"parameter missing: {param}")
            control_pars.append(kwargs[param])

        self.ode.set_parameters(tuple(control_pars))

    def integrate(
        self, t: float | np.floating[Any]
    ) -> tuple[
        Array1D[np.floating[Any]],
        Array1D[np.floating[Any]] | None,
        Array1D[np.floating[Any]] | None,
    ]:
        if self.nlyapunov:
            return self.ode.integrate(t)
        else:
            return self.ode.integrate(t), None, None


# }}}


# {{{ target


def make_input_variable(n: int | tuple[int, ...], offset: int = 0) -> JiTCXDEExpression:
    import jitcode

    y = np.empty(n, dtype=object)
    for i, idx in enumerate(np.ndindex(y.shape)):
        y[idx] = jitcode.y(offset + i)

    return y


def initialize_jitcode(
    f: Array1D[Any],
    y: Array1D[Any],
    *,
    nlyapunov: int = 0,
    control_pars: Sequence[sp.Symbol],
    module_location: pathlib.Path | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> jitcode.jitcode:
    import jitcode

    if nlyapunov > 0:
        return jitcode.jitcode_lyap(
            f,
            n=y.size,
            n_lyap=nlyapunov,
            verbose=verbose,
            control_pars=control_pars,
            module_location=str(module_location) if module_location else None,
        )
    else:
        return jitcode.jitcode(
            f,
            n=y.size,
            verbose=verbose,
            control_pars=control_pars,
            module_location=str(module_location) if module_location else None,
        )


def compile_jitcode(
    ode: jitcode.jitcode,
    *,
    module_location: pathlib.Path | None = None,
    simplify: bool = False,
    debug: bool = False,
    openmp: bool = False,
    verbose: bool = False,
) -> None:
    import time

    t_start = time.time()
    ode.compile_C(
        # FIXME: jitcode assumes lists
        extra_compile_args=list(cflags(debug=debug)),
        extra_link_args=list(linker_flags(debug=debug)),
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


def reload_jitcode(
    ode: jitcode.jitcode,
    module_location: pathlib.Path | None = None,
) -> None:
    if module_location is None:
        return

    if not module_location.exists():
        raise FileNotFoundError(module_location)

    from jitcxde_common.modules import find_and_load_module

    # FIXME: this is not exactly documented API
    try:
        ode.jitced = find_and_load_module(
            module_location.stem, str(module_location.parent)
        )
    except Exception as exc:
        log.error("Failed to reload module: %s.", exc, exc_info=exc)
    assert ode.jitced is not None

    ode.f = ode.jitced.f
    if hasattr(ode.jitced, "jac"):
        ode.jac = ode.jitced.jac

    ode._initialise = ode.jitced.initialise
    ode.compile_attempt = True


@dataclass(frozen=True)
class JiTCODETarget(JiTCXDETarget):
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
    def has_jitcode() -> bool:
        return has_jitcode()

    def initialize_module(
        self,
        f: Array1D[Any],
        y: Array1D[Any],
        *,
        control_pars: Sequence[sp.Symbol],
        module_location: pathlib.Path | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> jitcode.jitcode:
        return initialize_jitcode(
            f,
            y,
            nlyapunov=self.nlyapunov,
            control_pars=control_pars,
            module_location=module_location,
            verbose=verbose,
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
        import jitcode

        assert isinstance(de, jitcode.jitcode)

        compile_jitcode(
            de,
            module_location=module_location,
            simplify=simplify,
            debug=debug,
            openmp=openmp,
            verbose=verbose,
        )

    def reload_module(  # noqa: PLR6301
        self,
        de: jitcxde,
        module_location: pathlib.Path | None = None,
    ) -> None:
        import jitcode

        assert isinstance(de, jitcode.jitcode)
        reload_jitcode(de, module_location)

    def compile(
        self,
        code: Code,
        *,
        method: str = "RK45",
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
    ) -> JiTCODECompiledCode:
        import jitcode
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
        # FIXME: this will recompute the symbolic parameters
        parameters = fill_symbolic_parameters(code, parameters)
        func = self.lambdify(code, parameters=parameters)

        # evaluate to obtain symengine expressions
        y = make_input_variable(inputs.size)
        f = func(jitcode.t, y)

        # get control parameters, if any
        control_pars = tuple(
            param for param in parameters.values() if isinstance(param, sp.Symbol)
        )

        if module_location and module_location.exists():
            ode = self.initialize_module(
                f,
                y,
                control_pars=control_pars,
                verbose=verbose,
                module_location=module_location,
            )
        else:
            ode = self.initialize_module(
                f,
                y,
                control_pars=control_pars,
                verbose=verbose,
            )
            self.compile_module(
                ode,
                module_location=module_location,
                simplify=simplify,
                debug=debug,
                openmp=openmp,
                verbose=verbose,
            )

            if module_location is not None and module_location.exists():
                self.reload_module(ode)

        # NOTE: we cannot add parameters here because JiTCODE will try to compile
        # things and it won't fine the initial conditions.. it's up to the user.
        ode.set_integrator(method, atol=atol, rtol=rtol)

        return JiTCODECompiledCode(
            f=f,
            y=y,
            parameters=code.parameters,
            nlyapunov=self.nlyapunov,
            module_location=module_location,
            ode=ode,
            integrator_name=method,
            integrator_params={"atol": atol, "rtol": rtol},
        )


# }}}
