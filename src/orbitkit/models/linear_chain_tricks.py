# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import pymbolic.primitives as prim
from pymbolic.typing import Expression

import orbitkit.symbolic.primitives as sym
from orbitkit.symbolic.mappers import IdentityMapper
from orbitkit.typing import Array
from orbitkit.utils import module_logger

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from pytools import UniqueNameGenerator

log = module_logger(__name__)


# {{{ apply


class DiracDelayDistributor(IdentityMapper):
    """Distributes the Dirac delay kernel over a given expression graph.

    Note that this is done symbolically by applying the kernel to all variables.
    This is generally correct, since the Dirac kernel distributes insides functions
    and other operations directly.
    """

    kernel: sym.DiracDelayKernel
    """The kernel to distributed over the expression."""
    time: prim.Variable | None
    r"""The name of the time variable in the expression, if any. If provided, every
    instance of :math:`t` is directly replaced by :math:`t - \tau`.
    """
    inputs: set[prim.Variable] | None
    """A set of input variables over which to distribute the kernel. If given,
    any variables not in this set are assumed to be constants, so that the Dirac
    kernel has no effect.
    """

    def __init__(
        self,
        tau: sym.Expression,
        time: prim.Variable | None = None,
        inputs: Collection[prim.Variable] | None = None,
    ) -> None:
        self.kernel = sym.DiracDelayKernel(tau)
        self.time = time
        self.inputs = set(inputs) if inputs is not None else inputs

    def map_call(self, expr: prim.Call) -> Expression:
        # NOTE: do not allow other kernels in the expression
        # FIXME: this should be perfectly fine, but needs more work in
        # DelayKernelReplacer as well to include extra variables.
        func = expr.function
        if isinstance(func, sym.DelayKernel):
            raise ValueError(f"cannot distribute over expression: {expr}")

        return super().map_call(expr)

    def map_variable(self, expr: prim.Variable) -> Expression:
        # NOTE: we handle the following cases:
        # 1. If we hit the "time" variable, just set it to `t - tau`.
        # 2. If we hit one of the inputs, apply the kernel to it.
        # 3. If we hit another variable, leave it alone, i.e. assume constant.

        if self.time is not None and expr == self.time:
            return expr - self.kernel.avg

        if self.inputs is None:
            return self.kernel(expr)

        if expr in self.inputs:
            return self.kernel(expr)
        else:
            # NOTE: we assume that all our delay kernels are probability
            # distributions, so they sum up to 1 if we apply them to a constant
            return expr


class DelayKernelReplacer(IdentityMapper):
    """Replace delay kernels in the expression with constant (or no) delay variables.

    See :func:`transform_delay_kernels`.
    """

    time: prim.Variable | None
    """The name of the time variable in the expression, if any."""
    inputs: set[prim.Variable] | None
    """A set of input variables over in the expression. If given, any variables
    not in this set are assumed to be constants.
    """

    kernel_to_var_replace: dict[prim.Call, sym.Expression]
    """A mapping from delay kernel calls to the variables that replaced them. This
    attribute is mainly used as a cache for deduplication.
    """
    var_to_eqs: dict[sym.Expression, dict[str, sym.Expression]]
    r"""A mapping from new variables to a set of equations required to solve
    for those new variables, of the form :math:`\dot{z}_k = f_k(t, z)`. This
    set of equations will generally contain one equation for each variables.
    """

    unique_name_generator: UniqueNameGenerator
    """A unique name generator for new variables. This class reserves the prefix
    ``_ok_dde_chain_`` for its variables.
    """

    def __init__(
        self,
        time: prim.Variable | None = None,
        inputs: Collection[prim.Variable] | None = None,
    ) -> None:
        from pytools import UniqueNameGenerator

        self.time = time
        self.inputs = set(inputs) if inputs is not None else inputs

        self.kernel_to_var_replace = {}
        self.var_to_eqs = {}
        self.unique_name_generator = UniqueNameGenerator(forced_prefix="_ok_dde_chain_")

    def map_call(self, expr: prim.Call) -> Expression:
        func = expr.function

        if isinstance(func, sym.DelayKernel):
            if len(expr.parameters) != 1:
                raise ValueError(f"expected only one parameter in {func} call")

            try:
                return self.kernel_to_var_replace[expr]
            except KeyError:
                (param,) = expr.parameters
                assert isinstance(param, sym.Expression)

                suffix = param.name if isinstance(param, prim.Variable) else ""
                z = prim.Variable(self.unique_name_generator(suffix))

                if isinstance(func, sym.DiracDelayKernel):
                    z, equations = transform_dirac_delay_kernel(
                        func, param, z, time=self.time, inputs=self.inputs
                    )
                elif isinstance(func, sym.UniformDelayKernel):
                    z, equations = transform_uniform_delay_kernel(
                        func, param, z, time=self.time, inputs=self.inputs
                    )
                elif isinstance(func, sym.TriangularDelayKernel):
                    z, equations = transform_triangular_delay_kernel(
                        func, param, z, time=self.time, inputs=self.inputs
                    )
                elif isinstance(func, sym.GammaDelayKernel):
                    z, equations = transform_gamma_delay_kernel(
                        func, param, z, time=self.time, inputs=self.inputs
                    )
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
    *,
    time: prim.Variable | None = None,
    inputs: Collection[prim.Variable] | None = None,
) -> tuple[sym.Expression, Mapping[str, sym.Expression]]: ...


@overload
def transform_delay_kernels(
    expr: tuple[sym.Expression, ...],
    *,
    time: prim.Variable | None = None,
    inputs: Collection[prim.Variable] | None = None,
) -> tuple[tuple[sym.Expression, ...], Mapping[str, sym.Expression]]: ...


def transform_delay_kernels(
    expr: sym.Expression | tuple[sym.Expression, ...],
    *,
    time: prim.Variable | None = None,
    inputs: Collection[prim.Variable] | None = None,
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

    mapper = DelayKernelReplacer(time, inputs)
    expr = mapper(expr)  # ty: ignore[invalid-assignment]

    alleqs = {}
    for eqs in mapper.var_to_eqs.values():
        assert not any(name in alleqs for name in eqs)
        alleqs.update(eqs)

    return expr, constantdict(alleqs)


# }}}


# {{{ linear chain tricks


def transform_dirac_delay_kernel(
    kernel: sym.DiracDelayKernel,
    expr: sym.Expression,
    z: prim.Variable,
    *,
    time: prim.Variable | None = None,
    inputs: Collection[prim.Variable] | None = None,
) -> tuple[sym.Expression, dict[str, sym.Expression]]:
    """Transform the Dirac kernel applied to the given *expr*.

    The Dirac kernel is a limit case, where we simply distribute it over the
    variables in the given *expr* and return the result. No additional equations
    are required.
    """
    expr = DiracDelayDistributor(kernel.tau, time, inputs)(expr)  # ty: ignore[invalid-assignment]

    return expr, {}


def transform_uniform_delay_kernel(
    kernel: sym.UniformDelayKernel,
    expr: sym.Expression,
    z: prim.Variable,
    *,
    time: prim.Variable | None = None,
    inputs: Collection[prim.Variable] | None = None,
) -> tuple[sym.Expression, dict[str, sym.Expression]]:
    r"""Transform the uniform kernel into additional delay differential equation.

    .. math::

        \dot{z} = \frac{1}{2 \epsilon \tau} (
            expr(t - (1 - \epsilon) \tau) - expr(t - (1 + \epsilon) \tau)
        ).

    :returns: a mapping of variable names to equations. One of these variable
        names is the provided *z* variable and others can be derived from it.
    """
    epsilon, tau = kernel.epsilon, kernel.tau

    def dirac(tau: sym.Expression, expr: sym.Expression):
        return DiracDelayDistributor(tau, time=time, inputs=inputs)(expr)

    return z, {
        z.name: (
            (dirac((1 - epsilon) * tau, expr) - dirac((1 + epsilon) * tau, expr))
            / (2 * epsilon * tau)
        )
    }


def transform_triangular_delay_kernel(
    kernel: sym.TriangularDelayKernel,
    expr: sym.Expression,
    z: prim.Variable,
    *,
    time: prim.Variable | None = None,
    inputs: Collection[prim.Variable] | None = None,
) -> tuple[sym.Expression, dict[str, sym.Expression]]:
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

    def dirac(tau: sym.Expression, expr: sym.Expression):
        return DiracDelayDistributor(tau, time=time, inputs=inputs)(expr)

    w = prim.Variable(f"{z.name}_tr")
    return z, {
        z.name: w / (epsilon * tau) ** 2,
        w.name: (
            dirac((1 - epsilon) * tau, expr)
            - 2 * dirac(tau, expr)
            + dirac((1 + epsilon) * tau, expr)
        ),
    }


def transform_gamma_delay_kernel(
    kernel: sym.GammaDelayKernel,
    expr: sym.Expression,
    z: prim.Variable,
    *,
    time: prim.Variable | None = None,
    inputs: Collection[prim.Variable] | None = None,
) -> tuple[sym.Expression, dict[str, sym.Expression]]:
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
        return z, {z.name: alpha * (expr - z)}
    elif isinstance(p, int):
        zs = (z, *(prim.Variable(f"{z.name}_g{p}_{k}") for k in range(p - 1)))

        return z, {
            **{zs[k].name: alpha * (zs[k + 1] - zs[k]) for k in range(p - 1)},
            zs[p - 1].name: alpha * (expr - zs[p - 1]),
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
        return A @ ws - y

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
