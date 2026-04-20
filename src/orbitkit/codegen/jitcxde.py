# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

import numpy as np

import orbitkit.symbolic.primitives as sym
from orbitkit.codegen import Assignment, Code
from orbitkit.codegen.numpy import NumpyCodeGenerator, NumpyTarget
from orbitkit.typing import Array1D, ArrayND
from orbitkit.utils import module_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import symengine as sp
    from jitcxde_common import jitcxde

log = module_logger(__name__)

JiTCXDEExpression: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[Any]]
"""Array of expressions used by JiTC*DE code generation."""


def has_jitcode() -> bool:
    try:
        import jitcode  # noqa: F401
    except ImportError:
        return False
    else:
        return True


def has_jitcdde() -> bool:
    try:
        import jitcdde  # noqa: F401
    except ImportError:
        return False
    else:
        return True


# {{{ compilation flags

if platform.system() == "Windows":
    JITCXDE_COMMON_CFLAGS = (
        "/std:c11",
        "/arch:AVX2",
        "/wd4068",  # unknown pragma
        "/wd4146",  # unary minus operator applied to unsigned type
        "/wd4018",  # signed and unsigned comparison
    )

    JITCXDE_SYSTEM_RELEASE_CFLAGS = (
        "/O2",
        "/Oi",  # enable intrinsic functions
        "/Gy",  # dead code elimination at link time
        "/fp:fast",
        "/debug:none",
    )

    JITCXDE_SYSTEM_DEBUG_CFLAGS = (
        "/Od",
        "/Z7",
        "/GS",  # buffer security check
        "/sdl",  # additional checks
        "/analyze",  # enable static analysis
        "/fsanitize=address",  # requires MSVC 2019+
        "/RTC1",
    )

    JITCXDE_SYSTEM_RELEASE_LINKER_FLAGS = ("/ignore:4197",)
    JITCXDE_SYSTEM_DEBUG_LINKER_FLAGS = ("/ignore:4197",)
else:
    JITCXDE_COMMON_CFLAGS = (
        "-std=c11",
        "-march=native",
        "-mtune=native",
        "-Wno-unknown-pragmas",
    )

    JITCXDE_SYSTEM_DEBUG_CFLAGS = (
        # FIXME: jitcdde has some warnings here that are very noise. Should fix
        # there before enabling these in debug runs.
        # "-Wall",
        # "-Wextra",
        "-O0",
        "-ggdb",
    )

    JITCXDE_SYSTEM_RELEASE_CFLAGS = (
        # FIXME: -O3 and -ffast-math is not exactly safe. We should update our own
        # code generation and check if this actually makes things better.
        "-O3",
        "-ffast-math",
        # NOTE: this seemed to cause some issues with points near bifurcations, so
        # it's turned off by default for now.
        "-fno-associative-math",
        # NOTE: this seems to cause some invalid-writes or straight-up leaks in the
        # jitcdde C template. We disable it for now for safety.
        "-mno-avx512f",
        "-g0",
    )

    JITCXDE_SYSTEM_RELEASE_LINKER_FLAGS = ("-lm",)
    JITCXDE_SYSTEM_DEBUG_LINKER_FLAGS = ("-lm",)

JITCXDE_RELEASE_FLAGS = (*JITCXDE_COMMON_CFLAGS, *JITCXDE_SYSTEM_RELEASE_CFLAGS)
"""Compiler flags used for release builds of JiTC*DE modules."""

JITCXDE_DEBUG_FLAGS = (*JITCXDE_COMMON_CFLAGS, *JITCXDE_SYSTEM_DEBUG_CFLAGS)
"""Compiler flags used for debug builds of JiTC*DE modules."""

JITCXDE_RELEASE_LINKER_FLAGS = JITCXDE_SYSTEM_RELEASE_LINKER_FLAGS
"""Additional linker flags used for release builds of JiTC*DE modules."""

JITCXDE_DEBUG_LINKER_FLAGS = JITCXDE_SYSTEM_DEBUG_LINKER_FLAGS
"""Additional linker flags used for debug builds of JiTC*DE modules."""


def cflags(*, debug: bool = False) -> tuple[str, ...]:
    """Default compilation flags."""
    return JITCXDE_DEBUG_FLAGS if debug else JITCXDE_RELEASE_FLAGS


def linker_flags(*, debug: bool = False) -> tuple[str, ...]:
    """Default linker flags."""
    return JITCXDE_DEBUG_LINKER_FLAGS if debug else JITCXDE_RELEASE_LINKER_FLAGS


# }}}


# {{{ target


def fill_symbolic_parameters(
    code: Code,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if parameters is None:
        parameters = {}

    result = dict(parameters)
    if code.parameters:
        import symengine as sp

        for param in code.parameters:
            if param not in result:
                result[param] = sp.Symbol(param, real=True)

    return result


@dataclass(frozen=True)
class JiTCODECodeGenerator(NumpyCodeGenerator):
    sym_module: str = "sp"

    def map_function(self, expr: sym.Function, enclosing_prec: int) -> str:
        return f"vectorized({self.sym_module}.{expr.name})"


@dataclass(frozen=True)
class JiTCXDECompiledCode(ABC):
    """A cached compilation of JiTCXDE modules."""

    code: Code
    """The code and parameters that was compiled for this module."""
    module_location: pathlib.Path | None
    """The location of the compiled module."""

    f: Array1D[Any]
    """The symbolic SymEngine expression for the right-hand side."""
    y: Array1D[Any]
    """The symbolic SymEngine input variables."""

    nlyapunov: int
    """The number of Lyapunov exponents this module is computing. If zero, this
    is not a Lyapunov solver, so do not expect them as output.
    """

    @abstractmethod
    def set_initial_conditions(
        self,
        y: Array1D[np.floating[Any]],
        t: float = 0.0,
    ) -> None:
        """Set the initial conditions for the time evolution.

        The exact meaning of the initial conditions depends on the equation type:
        * For ODEs, this is just the initial conditions at time *t*.
        * For DDEs, this is a constant past up to and including time *t*.
        """

    @abstractmethod
    def set_parameters(self, *args: Any) -> None:
        """Set all symbolic parameters used by the solve."""

    @abstractmethod
    def integrate(
        self, t: float | np.floating[Any]
    ) -> tuple[
        Array1D[np.floating[Any]],
        Array1D[np.floating[Any]] | None,
        Array1D[np.floating[Any]] | None,
    ]:
        """Integrate the system to time *t*.

        This function has two return types:
        * If :attr:`nlyapunov` is non-zero, then it will return ``(y, lyap, w)``,
          where ``lyap`` is a local Lyapunov exponent and ``w`` is the corresponding
          weight on the current interval.
        * Otherwise, it just returns ``(y, None, None)``, so the last two values
          can be ignored.

        For more information on the exact return values, see the corresponding
        :mod:`jitcode` or `jitcdde` documentation.
        """


@dataclass(frozen=True)
class JiTCXDETarget(NumpyTarget, ABC):
    module: ClassVar[str] = "np"
    sym_module: ClassVar[str] = "sp"
    funcname: ClassVar[str] = "_lambdify_generated_func_jitcxde_symengine"

    # {{{ base class

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
            raise NotImplementedError(
                f"{type(self).__name__} cannot generate individual functions"
            )

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
    ) -> Callable[..., ArrayND[np.floating[Any]]]:
        # NOTE: if we need extra parameters, just add them as symbols. These
        # will be added properly according to JiTCODE in the compile function.
        parameters = fill_symbolic_parameters(code, parameters)
        return super().lambdify(code, parameters=parameters)

    # }}}

    # {{{ compilation

    @abstractmethod
    def initialize_module(
        self,
        f: Array1D[Any],
        y: Array1D[Any],
        *,
        control_pars: Sequence[sp.Symbol],
        module_location: pathlib.Path | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> jitcxde:
        """Initialize a JiTCXDE module with the given parameters.

        This function is meant to just create a JiTCXDE object and ensure that
        it is correctly initialized.
        """

    @abstractmethod
    def compile_module(
        self,
        de: jitcxde,
        *,
        module_location: pathlib.Path | None = None,
        simplify: bool = False,
        debug: bool = False,
        openmp: bool = False,
        verbose: bool = False,
    ) -> None:
        """Ensure that the *de* module is compiled.

        If *module_location* is given, the module should be compiled to this
        location, if it is not already. If it is, then all the additional flags
        are ignored and the previously compiled version will be reused.

        :arg simplify: apply simplifications to the right-hand side symbolic
            expressions with default tolerances in :mod:`sympy`.
        :arg debug: use debug compilation and linker flags.
        :arg openmp: enabled OpenMP pragmas and other compilation flags. This
            should not be turned on unless the system is very larger, since it
            will generally not result in a speedup (due to the additional thread
            setup costs).
        :arg verbose: show compilation messages and other logging information.
        """

    @abstractmethod
    def reload_module(self, de: jitcxde) -> None:
        """Reload a JiTCXDE module from disk."""

    # }}}


# }}}
