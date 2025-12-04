# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING

import pymbolic.primitives as prim
from pymbolic.typing import Expression

import orbitkit.models.symbolic as sym
from orbitkit.typing import Array
from orbitkit.utils import module_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools import UniqueNameGenerator

log = module_logger(__name__)


# {{{ apply


class DelayKernelReplacer(sym.IdentityMapper):
    equations: dict[str, sym.Expression]
    unique_name_generator: UniqueNameGenerator

    def __init__(self) -> None:
        from pytools import UniqueNameGenerator

        self.equations = {}
        self.unique_name_generator = UniqueNameGenerator()

    def map_call(self, expr: prim.Call) -> Expression:
        func = expr.function

        if isinstance(func, sym.DelayKernel):
            # NOTE: we're naming variables like `z_dirac_n`
            prefix = type(func).__name__[:-6].lower()
            z = prim.Variable(self.unique_name_generator(f"z_{prefix}"))

            (y,) = expr.parameters
            assert isinstance(y, prim.Variable)

            if isinstance(func, sym.DiracKernel):
                return sym.var(y.name, func.tau)
            elif isinstance(func, sym.UniformKernel):
                self.equations.update(transform_uniform_kernel(func, y, z.name))
            elif isinstance(func, sym.TriangularKernel):
                self.equations.update(transform_triangular_kernel(func, y, z.name))
            elif isinstance(func, sym.GammaKernel):
                self.equations.update(transform_gamma_kernel(func, y, z.name))
            else:
                raise TypeError(f"unsupported delay kernel: {type(func)}")

            return z
        else:
            return super().map_call(expr)


def transform_delay_kernels(expr: Array) -> tuple[Array, Mapping[str, sym.Expression]]:
    """Replace all distributed delay kernels with additional differential equations.

    The transformations can be found in [Macdonald2013]_. The supported kernels are
    * Gamma kernel: transformation using the standard linear chain trick into
      :math:`p` additional ODEs for each equation in *expr*.
    * Uniform kernel: transforms into an additional DDE for each variable in *expr*.
    * Triangular kernel: transforms into two additional DDEs for each variable
      in *expr*.
    * Dirac: already a convenient DDE.

    .. [Macdonald2013] N. MacDonald,
        *Time Lags in Biological Models*,
        Springer, 2013.

    :returns: a tuple of ``(expr, equations)``, where ``expr`` is the input
        expression with all delay kernels replaced by additional variables and
        ``equations`` is a mapping from variable names to right-hand side
        expressions.
    """
    from constantdict import constantdict

    mapper = DelayKernelReplacer()
    expr = mapper(expr)  # type: ignore[assignment,arg-type]

    return expr, constantdict(mapper.equations)


# }}}


# {{{ linear chain tricks


def transform_uniform_kernel(
    kernel: sym.UniformKernel,
    y: prim.Variable,
    replacement: str,
) -> dict[str, sym.Expression]:
    r"""Transform the uniform kernel into additional delay differential equations.

    .. math::

        \dot{z} = \frac{1}{2 \epsilon \tau} (
            y(t - (1 - \epsilon) \tau) - y(t - (1 + \epsilon) \tau)
        ).

    :returns: a mapping of variable names to equations. One of these variable
        names is the provided *replacement* name and other can be derived from it.
    """
    epsilon, tau = kernel.epsilon, kernel.tau
    z = replacement

    return {
        z: (sym.var(y.name, (1 - epsilon) * tau) - sym.var(y.name, (1 + epsilon) * tau))
        / (2 * epsilon * tau)
    }


def transform_triangular_kernel(
    kernel: sym.TriangularKernel,
    y: prim.Variable,
    replacement: str,
) -> dict[str, sym.Expression]:
    r"""Transform the triangular kernel into additional delay differential equations.

    .. math::

        \begin{aligned}
        \dot{z} & = \frac{w}{(\epsilon \tau)^2}, \\
        \dot{w} & =
            y(t - (1 - \epsilon) \tau)
            - 2 y(t - \tau)
            + y(t - (1 + \epsilon) \tau).
        \end{aligned}

    :returns: a mapping of variable names to equations. One of these variable
        names is the provided *replacement* name and other can be derived from it.
    """
    epsilon, tau = kernel.epsilon, kernel.tau
    z = replacement
    w = f"{replacement}_0"
    return {
        z: sym.var(w) / (epsilon * tau) ** 2,
        w: (
            sym.var(y.name, (1 - epsilon) * tau)
            - 2 * sym.var(y.name, tau)
            + sym.var(y.name, (1 + epsilon) * tau)
        ),
    }


def transform_gamma_kernel(
    kernel: sym.GammaKernel,
    y: prim.Variable,
    replacement: str,
) -> dict[str, sym.Expression]:
    r"""Transform the Gamma kernel into additional ordinary differential equations.

    .. math::

        \begin{aligned}
        \dot{z}_p & = \alpha (z_{p - 1} - z_p), \\
        \dots & \\
        \dot{z}_1 & = \alpha (y - z_1).
        \end{aligned}

    :returns: a mapping of variable names to equations. One of these variable
        names is the provided *replacement* name and other can be derived from it.
    """
    p, alpha = kernel.p, kernel.alpha

    if p == 1:
        z = replacement
        return {z: alpha * (y - sym.var(z))}
    elif isinstance(p, int):
        z = replacement
        zs = (replacement, *(f"{replacement}_{k}" for k in range(p - 1)))

        return {
            **{
                zs[k]: alpha * (sym.var(zs[k - 1]) - sym.var(zs[k]))
                for k in range(1, p)
            },
            zs[p - 1]: alpha * (y - sym.var(zs[p - 1])),
        }
    else:
        raise NotImplementedError(
            f"linear chain trick for Gamma kernel of order p = {p!r}"
        )


# }}}
