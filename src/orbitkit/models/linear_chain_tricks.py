# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, overload

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

Variable: TypeAlias = "sym.Variable | sym.CallDelay"


class DelayKernelReplacer(IdentityMapper):
    kernel_to_var_replace: dict[prim.Call, Variable]
    var_to_eqs: dict[Variable, dict[str, sym.Expression]]
    unique_name_generator: UniqueNameGenerator

    def __init__(self) -> None:
        from pytools import UniqueNameGenerator

        self.kernel_to_var_replace = {}
        self.var_to_eqs = {}
        self.unique_name_generator = UniqueNameGenerator()

    def map_call(self, expr: prim.Call) -> Expression:
        func = expr.function

        if isinstance(func, sym.DelayKernel):
            try:
                return self.kernel_to_var_replace[expr]
            except KeyError:
                (y,) = expr.parameters
                assert isinstance(y, prim.Variable)

                # NOTE: we're naming variables like `_z_n`
                z = prim.Variable(self.unique_name_generator(f"_{y.name}"))

                if isinstance(func, sym.DiracDelayKernel):
                    z = sym.CallDelay(y, func.tau)  # type: ignore[assignment]
                    equations = {}
                elif isinstance(func, sym.UniformDelayKernel):
                    equations = transform_uniform_delay_kernel(func, y, z)
                elif isinstance(func, sym.TriangularDelayKernel):
                    equations = transform_triangular_delay_kernel(func, y, z)
                elif isinstance(func, sym.GammaDelayKernel):
                    equations = transform_gamma_delay_kernel(func, y, z)
                else:
                    raise TypeError(f"unsupported delay kernel: {type(func)}") from None

                self.kernel_to_var_replace[expr] = z
                self.var_to_eqs[z] = equations
                return z
        else:
            return super().map_call(expr)


@overload
def transform_delay_kernels(
    expr: sym.Expression,
) -> tuple[sym.Expression, Mapping[str, sym.Expression]]: ...


@overload
def transform_delay_kernels(
    expr: tuple[sym.Expression, ...],
) -> tuple[tuple[sym.Expression, ...], Mapping[str, sym.Expression]]: ...


def transform_delay_kernels(
    expr: sym.Expression | tuple[sym.Expression, ...],
) -> tuple[sym.Expression | tuple[sym.Expression, ...], Mapping[str, sym.Expression]]:
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
    expr = mapper(expr)  # type: ignore[assignment]

    alleqs = {}
    for eqs in mapper.var_to_eqs.values():
        assert not any(name in alleqs for name in eqs)
        alleqs.update(eqs)

    return expr, constantdict(alleqs)


# }}}


# {{{ linear chain tricks


def transform_uniform_delay_kernel(
    kernel: sym.UniformDelayKernel,
    y: prim.Variable,
    z: prim.Variable,
) -> dict[str, sym.Expression]:
    r"""Transform the uniform kernel into additional delay differential equation.

    .. math::

        \dot{z} = \frac{1}{2 \epsilon \tau} (
            y(t - (1 - \epsilon) \tau) - y(t - (1 + \epsilon) \tau)
        ).

    :returns: a mapping of variable names to equations. One of these variable
        names is the provided *z* variable and others can be derived from it.
    """
    epsilon, tau = kernel.epsilon, kernel.tau

    return {
        z.name: (
            sym.CallDelay(y, (1 - epsilon) * tau)
            - sym.CallDelay(y, (1 + epsilon) * tau)
        )
        / (2 * epsilon * tau)
    }


def transform_triangular_delay_kernel(
    kernel: sym.TriangularDelayKernel,
    y: prim.Variable,
    z: prim.Variable,
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
        names is the provided *z* variable and others can be derived from it.
    """
    epsilon, tau = kernel.epsilon, kernel.tau
    w = prim.Variable(f"{z.name}s")
    return {
        z.name: w / (epsilon * tau) ** 2,
        w.name: (
            sym.CallDelay(y, (1 - epsilon) * tau)
            - 2 * sym.CallDelay(y, tau)
            + sym.CallDelay(y, (1 + epsilon) * tau)
        ),
    }


def transform_gamma_delay_kernel(
    kernel: sym.GammaDelayKernel,
    y: prim.Variable,
    z: prim.Variable,
) -> dict[str, sym.Expression]:
    r"""Transform the Gamma kernel into additional ordinary differential equations.

    .. math::

        \begin{aligned}
        \dot{z}_p & = \alpha (z_{p - 1} - z_p), \\
        \vdots & \\
        \dot{z}_1 & = \alpha (y - z_1).
        \end{aligned}

    :returns: a mapping of variable names to equations. One of these variable
        names is the provided *z* variable and others can be derived from it.
    """
    p, alpha = kernel.p, kernel.alpha

    if p == 1:
        return {z.name: alpha * (y - z)}
    elif isinstance(p, int):
        zs = (z, *(prim.Variable(f"{z.name}s_{k}") for k in range(p - 1)))

        return {
            **{zs[k].name: alpha * (zs[k + 1] - zs[k]) for k in range(p - 1)},
            zs[p - 1].name: alpha * (y - zs[p - 1]),
        }
    else:
        raise NotImplementedError(
            f"linear chain trick for Gamma kernel of order p = {p!r}"
        )


# }}}


# {{{ soe_gamma


def optimal_soe_gamma_points(
    p: float,
    alpha: float,
    tstart: float = 0.0,
    tfinal: float | None = None,
    *,
    dt: float | None = None,
    rtol: float = 1.0e-8,
) -> Array:
    r"""Approximate a range over which to fit a sum of exponentials approximation.

    See :func:`soe_gamma_varpo` and :func:`soe_gamma_mpm` for uses of this function.

    :arg tstart: start of fit interval.
    :arg tfinal: end of fit interval. If not provided, a final value is approximated
        based on the inverse CDF of the Gamma distribution to a tolerance *rtol*.
    :arg dt: step size used to discretize the range. If not provided, a value
        is approximated based on the variance of the Gamma distribution.
    :returns: an array of points :math:`\{t_i\}` over which to best fit the
        Gamma kernel with parameters :math:`(p, \alpha)`.
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
        dt = min(sigma / 32.0, 1.0e-2 * (tfinal - tstart))

    if tstart >= tfinal:
        raise ValueError(f"'tstart' ({tstart}) > 'tfinal' ({tfinal})")

    return np.arange(tstart, tfinal, dt)


def soe_gamma_varpo(
    t: Array,
    p: float,
    alpha: float,
    *,
    n: int | None = None,
    atol: float = 1.0e-8,
) -> tuple[Array, Array]:
    r"""Create a sum of exponentials approximation of the
    :math:`\mathrm{Gamma}(t; p, \alpha)` kernel using Variable Projection.

    .. math::

        \mathrm{Gamma}(t; p, \alpha) =
            \frac{\alpha^p}{\Gamma(p)} t^{p - 1} e^{-\alpha t}.

    The Gamma kernel is approximated using a sum of exponentials as

    .. math::

        \mathrm{Gamma}(t; p, \alpha) \approx
            \sum_{k = 0}^{n - 1} w_i e^{\lambda_i t}

    over the provided interval. Note that, if *p* is 1, then no approximation
    is necessary and a single pair is returned, regardless of *n*.

    :arg t: points at which to fit the Gamma kernel.
    :arg p: shape parameter of the Gamma kernel.
    :arg alpha: rate parameter of the Gamma kernel.
    :arg n: number of exponentials in the approximation.
    :arg atol: desired tolerance of the approximation. If *n* is not given, this
        value is used to estimate the required number of exponentials needed.

    :returns: a tuple of ``(ws, lambdas)`` of weights and rates for the sum of
        exponentials approximation. Note that the weights *ws* are real, but not
        necessarily positive (they tend to zig-zag between positive and negative
        for larger *p*).
    """
    if p < 1:
        raise ValueError(f"shape parameter 'p' must be >= 1: {p}")

    if alpha <= 0:
        raise ValueError(f"rate parameter 'alpha' must be positive: {alpha}")

    if n is None:
        # NOTE: vaguely based on Belkin and Monzon (2005)
        #       https://doi.org/10.1016/j.acha.2005.01.003
        # FIXME: this does not seem to work very well in practice
        n = int((p + 1) * np.log(1.0 / atol) * np.log(t[-1]) / np.pi) + 1
        n = min(n, t.size - 1)

    if n <= 0:
        raise ValueError(f"number of terms 'n' must be positive: {n}")

    if n >= t.size:
        raise ValueError(
            f"number of terms 'n' cannot be larger than 't.size': {n} >= {t.size}"
        )

    from scipy.special import gamma

    if p == 1:
        return np.array([alpha]), np.array([-alpha])

    x = t
    y = alpha**p / gamma(p) * t ** (p - 1) * np.exp(-alpha * x)

    # {{{ perform fit with variable projection

    from scipy.optimize import least_squares

    def fit_weights(loglambdas: Array) -> tuple[Array, Array]:
        lambdas = np.exp(loglambdas)
        A = np.exp(-lambdas[None, :] * x[:, None])
        ws, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return A, ws

    def residuals(loglambdas: Array) -> Array:
        A, ws = fit_weights(loglambdas)
        return A @ ws - y  # type: ignore[no-any-return]

    peak = max((p - 1) / alpha, np.min(np.diff(x)))

    min_rate = 1.0 / x[-1]
    max_rate = 1.0 / (0.5 * peak)
    lambdas0 = np.logspace(np.log10(min_rate), np.log10(max_rate), n)

    # TODO: use approximation of Jacobian from [Kaufman1975]? Should give much
    # faster+better results than the default 2-point finite difference.
    result = least_squares(
        residuals, np.log(lambdas0), method="lm", ftol=atol, xtol=atol
    )

    lambdas = np.exp(result.x)
    _, ws = fit_weights(result.x)

    # }}}

    return ws, -lambdas


def soe_gamma_mpm(
    t: Array,
    p: float,
    alpha: float,
    *,
    n: int | None = None,
    atol: float = 1.0e-8,
) -> tuple[Array, Array]:
    r"""Create a sum of exponentials approximation of the
    :math:`\mathrm{Gamma}(t; p, \alpha)` kernel using the Matrix Pencil Method.

    See :func:`soe_gamma_varpo`. Note that the weights and rates are returned
    as complex numbers for this method. However, it is expected that the resulting
    sum of exponentials is still real (to floating point precision).

    .. note::

        This method gives significantly less exponentials compared to
        :func:`soe_gamma_varpo` at the cost of complex numbers. If this is
        acceptable, this method should be preferred.

    :returns: a tuple of ``(ws, lambdas)`` of weights and rates for the sum of
        exponentials approximation. Note that, even if real, the weights are not
        expected to be positive.
    """

    if p < 1:
        raise ValueError(f"shape parameter 'p' must be >= 1: {p}")

    if alpha <= 0:
        raise ValueError(f"rate parameter 'alpha' must be positive: {alpha}")

    if n is None:
        # NOTE: vaguely based on Belkin and Monzon (2005)
        #       https://doi.org/10.1016/j.acha.2005.01.003
        n = int((p + 1) * np.log(1.0 / atol) * np.log(t[-1]) / np.pi) + 1
        n = min(n, t.size - 1)

    if n <= 0:
        raise ValueError(f"number of terms 'n' must be positive: {n}")

    if n >= t.size:
        raise ValueError(
            f"number of terms 'n' cannot be larger than 't.size': {n} >= {t.size}"
        )

    from scipy.special import gamma

    if p == 1:
        return np.array([alpha]), np.array([-alpha])

    x = t
    y = alpha**p / gamma(p) * t ** (p - 1) * np.exp(-alpha * x)

    # {{{ perform fit with matrix pencil method

    import scipy.linalg as sla

    # construct Hankel matrix SVD
    L = y.size // 2
    H = sla.hankel(y[:L], y[L - 1 : -1])
    U, S, _ = sla.svd(H)

    # determine number of terms from singular values and tolerance
    S /= S[0]
    indices = np.where(atol < S)[0]
    N = indices.size

    # find lambdas: eigenvalues of U[:-1]^\dagger @ U[1:]
    # FIXME: t must be uniform, but we're not really requiring that here?
    Z = sla.pinv(U[:-1, :N]) @ U[1:, :N]
    poles = np.linalg.eigvals(Z)
    lambdas = -np.log(poles) / (x[1] - x[0])

    # find weights
    A = np.exp(-lambdas[None, :] * x[:, None])
    ws, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    # }}}

    return ws, -lambdas


# }}}


# {{{ pade_gamma


def pade_gamma(
    gamma_p: float,
    alpha: float,
    *,
    n: int | None = None,
    m: int = 6,
) -> tuple[Array, Array]:
    r"""Create a Padé approximant for the Laplace transform of the
    :math:`\mathrm{Gamma}(t; p, \alpha)` kernel.

    The Laplace transform of the Gamma kernel is given by

    .. math::

        \mathcal{L}\{\mathrm{Gamma}\}(s) =
            \left(\frac{\alpha}{\alpha + s}\right)^p.

    We seek an approximation of the form

    .. math::

        \frac{a_0 + a_1 x + \cdots + a_n x^n}
             {b_0 + b_1 x + \cdots + a_m x^m},

    for orders :math:`(m, n)`, i.e. a Padé approximant at :math:`s = 0`.

    This is an alternative approximation to the sum of exponentials (e.g.
    :func:`soe_gamma_mpm`) with different compromises. In particular, the Padé
    approximant of order :math:`n > 1` will exactly match the  higher-order
    moments of the Gamma distribution. However, it can result in kernels that
    have negative values and multiple local extrema in physical space (unlike
    the original Gamma kernel).

    Note that we construct the Padé approximant by performing a Taylor expansion
    at :math:`s = 0`. This Taylor expansion has a radius of convergence of only
    :math:`|s| < \alpha`. However, the approximant is expected to remain valid
    even outside this range and provide a good approximation of the Gamma
    function (but may require :math:`m > 1`).

    .. warning::

        This method seems to mostly work well for small orders :math:`p` of the
        Gamma kernel. Thss kind of negates any benefits it may have, since ideally
        we would use it for large :math:`p`.

    :arg n: order of the polynomial in the numerator.
    :arg m: order of the polynomial in the denominator.
    :returns: a tuple ``(p, q)`` of polynomial coefficients for the numerator
        and denominator. The coefficient order matches that of :mod:`numpy.polynomial`.
    """

    if n is None:
        # NOTE: this has the best chance of giving a good approximation for a
        # wider range of s values
        n = m - 1

    if n < 0 or not isinstance(n, int):
        raise ValueError(f"order 'n' must be a positive integer: {n}")

    if m < 0 or not isinstance(m, int):
        raise ValueError(f"order 'm' must be a positive integer: {m}")

    if m < n:
        raise ValueError(f"order 'm' cannot be smaller than 'n': {m} < {n}")

    # gather Taylor coefficients
    from scipy.special import binom

    k = np.arange(n + m + 1)
    if float(gamma_p).is_integer():
        taylor_coeffs = (-1.0) ** k * binom(gamma_p + k - 1, k) / alpha**k
    else:
        taylor_coeffs = binom(-gamma_p, k) / alpha**k

    # evaluate Pade approximant
    # NOTE: pade seems to return a `np.poly1d`, which has coefficients in a
    # reverse order to `np.polynomial` classes, so we flip them for compatibility
    # NOTE: this could cause issues: https://github.com/scipy/scipy/issues/20064
    from scipy.interpolate import pade

    ppoly, qpoly = pade(taylor_coeffs, m, n)
    # assert ppoly.order <= qpoly.order

    return np.flip(ppoly.coefficients), np.flip(qpoly.coefficients)


# }}}
