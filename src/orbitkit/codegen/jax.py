# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from orbitkit.codegen import Code, execute_code
from orbitkit.codegen.numpy import NumpyTarget
from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


@dataclass(frozen=True)
class JaxTarget(NumpyTarget):
    module: ClassVar[str] = "jnp"
    funcname: ClassVar[str] = "_lambdify_generated_func_jax"

    jit: bool = True
    """If *True*, the functions returned by :meth:`lambdify` will be JITed."""

    def _get_module(self) -> Any:  # noqa: PLR6301
        import jax.numpy as jnp

        return jnp

    def lambdify(
        self,
        code: Code,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> Callable[..., Array]:
        import jax

        func = execute_code(code)
        cargs = tuple(jax.device_put(arg) for arg in code.args)

        if code.parameters:
            if parameters is None:
                raise ValueError(f"missing parameters: {code.parameters}")

            params = []
            for name in code.parameters:
                if name not in parameters:
                    raise ValueError(f"missing parameter: '{name}'")

                params.append(jax.device_put(parameters[name]))

            cargs = (*cargs, *params)

        def wrapper(*args: Array) -> Array:
            return func(*args, *cargs)

        return jax.jit(wrapper) if self.jit else wrapper
