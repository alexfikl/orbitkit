# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pymbolic.primitives as prim
from pymbolic.typing import Expression

import orbitkit.symbolic.primitives as sym
from orbitkit.symbolic.mappers import IdentityMapper
from orbitkit.typing import Array
from orbitkit.utils import module_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools import UniqueNameGenerator

log = module_logger(__name__)


# {{{ apply


class DelayKernelReplacer(IdentityMapper):
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
            z = prim.Variable(self.unique_name_generator("_z"))

            (y,) = expr.parameters
            assert isinstance(y, prim.Variable)

            if isinstance(func, sym.DiracDelayKernel):
                return sym.var(y.name, func.tau)
            elif isinstance(func, sym.UniformDelayKernel):
                self.equations.update(transform_uniform_delay_kernel(func, y, z.name))
            elif isinstance(func, sym.TriangularDelayKernel):
                self.equations.update(
                    transform_triangular_delay_kernel(func, y, z.name)
                )
            elif isinstance(func, sym.GammaDelayKernel):
                self.equations.update(transform_gamma_delay_kernel(func, y, z.name))
            else:
                raise TypeError(f"unsupported delay kernel: {type(func)}")

            return z
        else:
            return super().map_call(expr)


def transform_delay_kernels(
    expr: sym.Expression | Array,
) -> tuple[sym.Expression | Array, Mapping[str, sym.Expression]]:
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


def transform_uniform_delay_kernel(
    kernel: sym.UniformDelayKernel,
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


def transform_triangular_delay_kernel(
    kernel: sym.TriangularDelayKernel,
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


def transform_gamma_delay_kernel(
    kernel: sym.GammaDelayKernel,
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
                zs[k]: alpha * (sym.var(zs[k + 1]) - sym.var(zs[k]))
                for k in range(p - 1)
            },
            zs[p - 1]: alpha * (y - sym.var(zs[p - 1])),
        }
    else:
        raise NotImplementedError(
            f"linear chain trick for Gamma kernel of order p = {p!r}"
        )


# }}}


# {{{ approximate_soe_gamma_kernel


def approximate_soe_range(
    p: float,
    alpha: float,
    tstart: float = 0.0,
    tfinal: float | None = None,
    *,
    dt: float | None,
    rtol: float = 1.0e-8,
) -> Array:
    """Approximate a range over which to fit a sum of exponentials approximation.

    See :func:`approximate_gamma_kernel`.
    """
    import scipy.stats as ss

    if not 0 < rtol < 1:
        raise ValueError(f"tolerance 'rtol' not in (0, 1): {rtol}")

    if tfinal is None:
        # NOTE: use the inverse CDF to get a good guess of when we're at tolerance
        tfinal = ss.gamma.ppf(1.0 - rtol, a=p, scale=1.0 / alpha)
        tfinal = max(tfinal, tstart + 1.0)

    if dt is None:
        # NOTE: this is the variance of the Gamma distribution
        sigma = np.sqrt(p) / alpha
        dt = min(sigma / 20.0, 1.0e-2 * (tfinal - tstart))

    if tstart >= tfinal:
        raise ValueError(f"'tstart' ({tstart}) > 'tfinal' ({tfinal})")

    return np.arange(tstart, tfinal, dt)


def approximate_soe_gamma_kernel(
    t: Array,
    p: float,
    alpha: float,
    *,
    n: int | None = None,
) -> tuple[Array, Array]:
    r"""Create a sum of exponentials approximation of the
    :math:`\mathrm{Gamma}(t; p, \alpha)` kernel.

    .. math::

        \mathrm{Gamma}(t; p, \alpha) =
            \frac{\alpha^p}{\Gamma(p)} t^{p - 1} e^{-\alpha t}.

    The Gamma kernel is approximated using a sum of exponentials as

    .. math::

        \mathrm{Gammma}(t; p, \alpha) \approx
            \sum_{k = 0}^{n - 1} w_i e^{\lambda_i t}

    over the provided interval.

    :arg t: time points at which to fit the Gamma kernel.
    :arg p: shape parameter of the Gamma kernel.
    :arg alpha: rate parameter of the Gamma kernel.
    :arg n: number of exponentials in the approximation.
    :returns: a tuple of ``(w, \lambda)`` of weights and rates for the sum of
        exponentials approximation.
    """
    if p <= 1:
        raise ValueError(f"shape parameter 'p' must be >= 1: {p}")

    if alpha <= 0:
        raise ValueError(f"rate parameter 'alpha' must be positive: {alpha}")

    if n is None:
        n = int(p) + 5

    if n <= 0:
        raise ValueError(f"number of terms 'n' must be positive: {n}")

    from scipy.special import gamma

    x = t
    y = alpha**p / gamma(p) * t ** (p - 1) * np.exp(-alpha * x)

    # {{{ perform fit with variable projection + least squares

    from scipy.optimize import least_squares

    def fit_weights(loglambdas: Array) -> tuple[Array, Array]:
        lambdas = np.exp(loglambdas)
        A = np.exp(-lambdas[None, :] * x[:, None])
        w, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return A, w

    def residuals(loglambdas: Array) -> Array:
        A, w = fit_weights(loglambdas)
        return A @ w - y  # type: ignore[no-any-return]

    peak = max((p - 1) / alpha, np.min(np.diff(x)))

    min_rate = 1.0 / x[-1]
    max_rate = 1.0 / (0.5 * peak)
    lambdas0 = np.logspace(np.log10(min_rate), np.log10(max_rate), n)

    result = least_squares(residuals, np.log(lambdas0), method="lm")

    lambdas = np.exp(result.x)
    _, w = fit_weights(result.x)

    # }}}

    return w, lambdas


# }}}
