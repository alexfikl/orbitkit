# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

import numpy as np

import orbitkit.symbolic.primitives as sym
from orbitkit.codegen.numpy import NumpyCodeGenerator, NumpyTarget
from orbitkit.utils import module_logger

log = module_logger(__name__)

JiTCXDEExpression: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[Any]]
"""Array of expressions used by JiTC*DE code generation."""


# {{{ compilation flags

if platform.system() == "Windows":
    JITCXDE_COMMON_CFLAGS = [
        "/std:c11",
        "/arch:AVX2",
        "/wd4068",  # unknown pragma
        "/wd4146",  # unary minus operator applied to unsigned type
        "/wd4018",  # signed and unsigned comparison
    ]

    JITCXDE_SYSTEM_RELEASE_CFLAGS = [
        "/O2",
        "/Oi",  # enable intrinsic functions
        "/Gy",  # dead code elimination at link time
        "/fp:fast",
        "/debug:none",
    ]

    JITCXDE_SYSTEM_DEBUG_CFLAGS = [
        "/Od",
        "/Z7",
        "/GS",  # buffer security check
        "/sdl",  # additional checks
        "/analyze",  # enable static analysis
        "/fsanitize=address",  # requires MSVC 2019+
        "/RTC1",
    ]

    JITCXDE_SYSTEM_RELEASE_LINKER_FLAGS = ["/ignore:4197"]
    JITCXDE_SYSTEM_DEBUG_LINKER_FLAGS = ["/ignore:4197"]
else:
    JITCXDE_COMMON_CFLAGS = [
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-march=native",
        "-mtune=native",
        "-Wno-unknown-pragmas",
    ]

    JITCODE_DEBUG_CFLAGS = [
        "-O0",
        "-ggdb",
    ]

    JITCXDE_SYSTEM_RELEASE_CFLAGS = [
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
    ]

    JITCXDE_SYSTEM_RELEASE_LINKER_FLAGS = ["-lm"]
    JITCXDE_SYSTEM_DEBUG_LINKER_FLAGS = ["-lm"]

JITCXDE_RELEASE_FLAGS = [*JITCXDE_COMMON_CFLAGS, *JITCXDE_SYSTEM_RELEASE_CFLAGS]
"""Compiler flags used for release builds of JiTC*DE modules."""

JITCXDE_DEBUG_FLAGS = [*JITCXDE_COMMON_CFLAGS, *JITCXDE_SYSTEM_DEBUG_CFLAGS]
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


@dataclass(frozen=True)
class JiTCODECodeGenerator(NumpyCodeGenerator):
    sym_module: str = "sp"

    def map_function(self, expr: sym.Function, enclosing_prec: int) -> str:
        return f"vectorized({self.sym_module}.{expr.name})"


@dataclass(frozen=True)
class JiTCXDETarget(NumpyTarget):
    module: ClassVar[str] = "np"
    sym_module: ClassVar[str] = "sp"
    funcname: ClassVar[str] = "_lambdify_generated_func_jitcxde_symengine"

    def _get_code_generator(self, inputs: set[str]) -> NumpyCodeGenerator:
        return JiTCODECodeGenerator(
            inputs=inputs, module=self.module, sym_module=self.sym_module
        )


# }}}
