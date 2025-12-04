# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Protocol, TypeAlias

import numpy as np
import pymbolic.primitives as prim
from pymbolic.mapper import IdentityMapper as IdentityMapperBase
from pymbolic.mapper.stringifier import PREC_NONE
from pymbolic.mapper.stringifier import StringifyMapper as StringifyMapperBase
from pymbolic.typing import Expression as PymbolicExpression

from orbitkit.typing import Array, DataclassInstanceT
from orbitkit.utils import module_logger

log = module_logger(__name__)

Expression: TypeAlias = (
    int | float | complex | np.inexact | np.integer | prim.ExpressionNode
)
"""A union of allowable classes that can be part of an expression."""
ArrayExpression: TypeAlias = Expression | np.ndarray[tuple[int, ...], np.dtype[Any]]
"""An expression that can also be an array of expressions."""

# {{{ dataclass as symbolic


def ds_symbolic(
    obj: DataclassInstanceT,
    *,
    rec: bool = False,
    rattrs: set[str] | None = None,
) -> DataclassInstanceT:
    """Fill in all the fields of *cls* with symbolic variables.

    :arg rec: if *True*, automatically recurse into all child dataclasses.
    :arg rattrs: a set of attribute names that will be recursed into regardless
        of the value of the *rec* flag.
    """
    from dataclasses import is_dataclass, replace

    if rattrs is None:
        rattrs = set()

    kwargs: dict[str, Any] = {}
    for f in fields(obj):
        attr = getattr(obj, f.name)
        if (rec or f.name in rattrs) and is_dataclass(attr):
            assert not isinstance(attr, type)
            kwargs[f.name] = ds_symbolic(attr, rec=rec, rattrs=rattrs)
            continue

        if isinstance(attr, tuple):
            kwargs[f.name] = tuple(
                prim.Variable(f"{f.name}_{i}") for i in range(len(attr))
            )
        elif isinstance(attr, np.ndarray):
            kwargs[f.name] = MatrixSymbol(f.name, attr.shape)
        else:
            kwargs[f.name] = prim.Variable(f.name)

    return replace(obj, **kwargs)


# }}}


# {{{ expressions


Variable = prim.Variable


@prim.expr_dataclass()
class ExpressionNode(prim.ExpressionNode):
    """A base class for ``orbitkit``-specific expression nodes."""

    def make_stringifier(  # noqa: PLR6301
        self,
        originating_stringifier: StringifyMapperBase[Any] | None = None,
    ) -> StringifyMapper:
        return StringifyMapper()


@prim.expr_dataclass()
class Contract(ExpressionNode):
    """Describes a tensor contraction (i.e. a sum along the given axes).

    .. note::

        Note that this does not check if the given expression evaluates to a
        tensor where all of these axes are valid.
    """

    aggregate: Expression
    axes: tuple[int, ...]
    """A tuple of axes to contract along."""


@prim.expr_dataclass()
class EinsteinSummation(ExpressionNode):
    """Describes a general Einstein summation between multiple terms.

    .. note::

        Note that this does not check if the given expression evaluates to a
        tensor where all of these indices are valid.
    """

    subscripts: str
    """Description of indices to sum. The indices must be a comma separated list
    for each of of the ``operands``. An optional ``->`` can be included to
    label the output indices.
    """
    operands: tuple[Expression, ...]


def einsum(subscripts: str, *operands: Expression) -> EinsteinSummation:
    """Construct a :class:`EinsteinSummation`."""

    if "->" in subscripts:
        inputs, _ = subscripts.split("->")
    else:
        inputs = subscripts

    if len(inputs.split(",")) != len(operands):
        raise ValueError(
            "'operands' do not match those provided in 'subscripts': "
            f"got {len(operands)} operands for '{subscripts}'"
        )

    return EinsteinSummation(subscripts, operands)


@prim.expr_dataclass()
class DotProduct(ExpressionNode):
    """Describes a standard dot product.

    .. note::

        This expression node does not check if the left and right operands evaluate
        to two tensors with appropriate dimensions for this dot product.

    .. warning::

        In most cases, this is assumed to have :func:`numpy.dot` semantics, i.e.
        it translates to an inner product for vectors, a matrix product for
        matrices, etc.
    """

    left: Array | Expression
    right: Array | Expression


@prim.expr_dataclass()
class Reshape(ExpressionNode):
    """Describes a reshaping operations on an n-dimensional array.

    .. note::

        This expression node does not check that the operand can be reshaped to
        the new shape. This will be done at code generation time.
    """

    aggregate: Expression
    shape: tuple[int, ...]
    """A tuple of integers describing the new shape. A single ``-1`` is allowed
    to expand the remaining dimensions."""

    @property
    def ndim(self) -> int:
        """Number of dimensions in the reshaped array."""
        return len(self.shape)


@prim.expr_dataclass()
class MatrixSymbol(prim.Variable):
    """A :class:`~pymbolic.primitives.Variable` that represents an
    n-dimensional array.
    """

    shape: tuple[int, ...]
    """The shape of the symbolic array."""

    @property
    def ndim(self) -> int:
        """Number of dimensions in the symbolic array."""
        return len(self.shape)

    def reshape(self, *shape: int) -> Reshape:
        """Reshape the array into the new *shape*."""

        known_shape = [d for d in shape if d != -1]
        if len(shape) - len(known_shape) > 1:
            raise ValueError(f"can only specify one unknown (-1) dimension: {shape}")

        size = np.prod(self.shape)
        new_size = np.prod(known_shape)
        has_one = len(known_shape) != len(shape)

        if (has_one and size % new_size != 0) or (not has_one and size != new_size):
            raise ValueError(f"cannot reshape array of size {size} into shape {shape}")

        return Reshape(self, shape)


@prim.expr_dataclass()
class VariableWithDelay(prim.Variable):
    """A variable with a constant delay, e.g. :math:`y(t - \tau)`."""

    tau: Expression
    """The expression for the delay. This is expected to evaluate to be
    convertible or evaluate to float.
    """


def var(name: str, tau: Expression | None = None) -> prim.Variable:
    """
    :arg name: name of the new variable.
    :arg tau: optional constant delay. If this is zero or *None*, it is assumed
        that there is no delay and a standard variable is returned.
    """
    if tau is None or tau == 0:
        return prim.Variable(name)
    else:
        return VariableWithDelay(name=name, tau=tau)


# }}}


# {{{ functions


@prim.expr_dataclass()
class Function(prim.Variable):
    r"""A known special function (e.g. :math:`\sin`, etc.)."""


sin = Function("sin")
"""The sine function."""
cos = Function("cos")
"""The cosine function."""
exp = Function("exp")
"""The exponential function."""
tanh = Function("tanh")
"""The hyperbolic tangent function."""
gamma = Function("gamma")
"""The Gamma function."""

# }}}


# {{{ delay kernels


@prim.expr_dataclass()
class DelayKernel(ExpressionNode):
    """A general delay kernel for distributed delay equations."""


@prim.expr_dataclass()
class DiracKernel(DelayKernel):
    r"""A delay kernel based on the Dirac distribution.

    .. math::

        \mathrm{Dirac}(t) = \delta(t)
    """

    tau: Expression
    """Average delay of the kernel."""


@prim.expr_dataclass()
class GammaKernel(DelayKernel):
    r"""A delay kernel based on the Gamma distribution.

    .. math::

        \mathrm{Gamma}(t; p, \alpha) =
            \frac{\alpha^p}{\Gamma(p)} t^{p - 1} e^{-\alpha t}.

    The average delay is given by :math:`\tau = p / \alpha`.
    """

    p: float
    """Shape parameter of the distribution."""
    alpha: Expression
    """Rate parameter of the distribution."""


@prim.expr_dataclass()
class UniformKernel(DelayKernel):
    r"""A delay kernel based on the uniform distribution.

    .. math::

        \mathrm{Uniform}(t; \epsilon, \tau) =
        \begin{cases}
        \dfrac{1}{2 \epsilon \tau}, &
            \quad (1 - \epsilon) \tau < t < (1 + \epsilon) \tau, \\
        0, & \quad \text{otherwise}.
        \end{cases}
    """

    epsilon: Expression
    """Parameter controlling the width of the plateau."""
    tau: Expression
    """Average delay of the kernel."""


@prim.expr_dataclass()
class TriangularKernel(DelayKernel):
    r"""A delay kernel based on the triangular distribution.

    .. math::

        \mathrm{Triangular}(t; \epsilon, \tau) =
        \begin{cases}
        \dfrac{t - (1 - \epsilon) \tau}{(\epsilon \tau)^2}, &
            \quad (1 - \epsilon) \tau < t < \tau, \\
        \dfrac{(1 + \epsilon) \tau - t}{(\epsilon \tau)^2}, &
            \quad \tau \le t < (1 + \epsilon) \tau, \\
        0, & \quad text{otherwise}.
        \end{cases}
    """

    epsilon: Expression
    """Parameter controlling the width of the base of the triangle."""
    tau: Expression
    """Average delay of the kernel."""


# }}}

# {{{ parametrized functions


class RateFunction(Protocol):
    """A callable protocol for rate functions used in many neuron models.

    .. automethod:: __call__
    """

    def __call__(self, V: Expression) -> Expression:
        """Evaluate the rate function for the given membrane potential."""


@dataclass(frozen=True)
class ExponentialRate:
    r"""Exponential rate function.

    .. math::

        f(V; a, \theta, \sigma) = a \exp\left(-\frac{(V - \theta)}{\sigma}\right).
    """

    a: Expression
    theta: Expression
    sigma: Expression

    def __call__(self, V: Expression) -> Expression:
        return self.a * exp(-(V - self.theta) / self.sigma)


@dataclass(frozen=True)
class SigmoidRate:
    r"""Sigmoid rate function.

    .. math::

        f(V; a, \theta, \sigma) =
            \frac{a}{1 + \exp\left(-\frac{(V - \theta)}{\sigma}\right)}.
    """

    a: Expression
    theta: Expression
    sigma: Expression

    def __call__(self, V: Expression) -> Expression:
        return self.a / (1.0 + exp(-(V - self.theta) / self.sigma))


@dataclass(frozen=True)
class Expm1Rate:
    r"""Reciprocal of the ``expm1`` function.

    .. math::

        f(V; a, \theta, \sigma) =
            \frac{a}{1 - \exp\left(-\frac{(V - \theta)}{\sigma}\right)}.
    """

    a: Expression
    theta: Expression
    sigma: Expression

    def __call__(self, V: Expression) -> Expression:
        return self.a / (1.0 - exp(-(V - self.theta) / self.sigma))


@dataclass(frozen=True)
class LinearExpm1Rate:
    r"""A linear exponential rate function.

    .. math::

        f(V; a, b, \theta, \sigma) =
            \frac{a V + b}{1 - \exp\left(-\frac{(V - \theta)}{\sigma}\right)}
    """

    a: Expression
    b: Expression
    theta: Expression
    sigma: Expression

    def __call__(self, V: Expression) -> Expression:
        return (self.a * V + self.b) / (1.0 - exp(-(V - self.theta) / self.sigma))


@dataclass(frozen=True)
class TanhRate:
    r"""A :math:`tanh` based rate function.

    .. math::

        f(V; A, \theta, \sigma) =
            A \left[1 + \tanh\left(\frac{V - \theta}{\sigma}\right)\right].
    """

    a: Expression
    theta: Expression
    sigma: Expression

    def __call__(self, V: Expression) -> Expression:
        return self.a * (1.0 + tanh((V - self.theta) / self.sigma))


# }}}


# {{{ mappers


class IdentityMapper(IdentityMapperBase[[]]):
    def map_contract(self, expr: Contract) -> PymbolicExpression:
        aggregate = self.rec_arith(expr.aggregate)
        if aggregate is expr.aggregate:
            return expr

        return type(expr)(aggregate=aggregate, axes=expr.axes)

    def map_einstein_summation(self, expr: EinsteinSummation) -> PymbolicExpression:
        operands = tuple(self.rec_arith(operand) for operand in expr.operands)
        if all(a is b for a, b in zip(operands, expr.operands, strict=True)):
            return expr

        return type(expr)(subscripts=expr.subscripts, operands=operands)

    def map_dot_product(self, expr: DotProduct) -> PymbolicExpression:
        # NOTE: mypy is upset because left/right can also be ndarrays
        left = self.rec_arith(expr.left)  # type: ignore[arg-type]
        right = self.rec_arith(expr.right)  # type: ignore[arg-type]
        if left is expr.left and right is expr.right:
            return expr

        return type(expr)(left, right)

    def map_reshape(self, expr: Reshape) -> PymbolicExpression:
        aggregate = self.rec_arith(expr.aggregate)
        if aggregate is expr.aggregate:
            return expr

        return type(expr)(aggregate=aggregate, shape=expr.shape)

    def map_variable_with_delay(self, expr: VariableWithDelay) -> PymbolicExpression:
        tau = self.rec_arith(expr.tau)
        if tau is expr.tau:
            return expr

        return type(expr)(expr.name, tau)

    # def map_delay_kernel(self, expr: DelayKernel) -> PymbolicExpression: ...


# }}}

# {{{ stringifier


class StringifyMapper(StringifyMapperBase[Any]):
    def map_variable(self, expr: prim.Variable, enclosing_prec: int) -> str:  # noqa: PLR6301
        from sympy.printing.pretty.pretty_symbology import pretty_symbol

        result = pretty_symbol(expr.name)
        return str(result)

    def map_variable_with_delay(
        self, expr: VariableWithDelay, enclosing_prec: int
    ) -> str:
        from sympy.printing.pretty.pretty_symbology import pretty_symbol

        name = pretty_symbol(expr.name)
        tau = self.rec(expr.tau, PREC_NONE)

        return f"{name}[delay={tau}]"

    def map_numpy_array(  # noqa: PLR6301
        self, expr: np.ndarray[tuple[int, ...], np.dtype[Any]], enclosing_prec: int
    ) -> str:
        return repr(expr)

    def map_contract(self, expr: Contract, enclosing_prec: int) -> str:
        aggregate = self.rec(expr.aggregate, PREC_NONE)
        return f"sum({aggregate}, axis={expr.axes})"

    def map_reshape(self, expr: Reshape, enclosing_prec: int) -> str:
        aggregate = self.rec(expr.aggregate, PREC_NONE)
        return f"({aggregate}).reshape{expr.shape}"

    def map_dot_product(self, expr: DotProduct, enclosing_prec: int) -> str:
        left = self.rec(expr.left, PREC_NONE)  # type: ignore[arg-type]
        right = self.rec(expr.right, PREC_NONE)  # type: ignore[arg-type]
        return f"dot({left}, {right})"

    def map_delay_kernel(self, expr: DelayKernel) -> str:  # noqa: PLR6301
        name = type(expr).__name__[:-6].lower()

        return f"{name}_knl"


def stringify(expr: Expression) -> str:
    return StringifyMapper()(expr)


# }}}

# {{{ model


@dataclass(frozen=True)
class Model(ABC):
    @property
    @abstractmethod
    def variables(self) -> tuple[str, ...]:
        """A tuple of all the state variables in the system."""

    @abstractmethod
    def evaluate(self, t: Expression, *args: MatrixSymbol) -> tuple[Expression, ...]:
        """
        :returns: an expression of the model evaluated at the given arguments.
        """

    def symbolify(
        self,
        n: int | tuple[int, ...],
        *,
        full: bool = False,
    ) -> tuple[tuple[prim.Variable, ...], tuple[Expression, ...]]:
        r"""Evaluate model on symbolic arguments for a specific size *n*.

        This function creates appropriate symbolic variables and calls
        :meth:`evaluate` to create the fully symbolic model. These can then
        essentially be passed directly to some other backend for code generation.

        :returns: a tuple of ``(args, model)``, where *args* are the symbolic
            variables (i.e. :class:`~pymbolic.primitives.Variable` and such) and
            the model is given as a tuple of symbolic expression (one for each
            input variable).
        """

        x = self.variables
        if isinstance(n, int):
            n = (n,) * len(x)

        if len(x) != len(n):
            raise ValueError(
                f"number of variables does not match sizes: variables {x} for sizes {n}"
            )

        if not all(n[0] == n_i for n_i in n[1:]):
            raise NotImplementedError(f"only uniform sizes are supported: {n}")

        t = prim.Variable("t")
        args = [MatrixSymbol(name, (n_i,)) for n_i, name in zip(n, x, strict=True)]

        model = self
        if full:
            model = ds_symbolic(model, rec=False, rattrs={"param"})

        return (t, *args), model.evaluate(t, *args)

    def __str__(self) -> str:
        n = getattr(self, "n", None)
        if n is None:
            n = 8

        args, exprs = self.symbolify(n, full=True)

        eqs = []
        for i, (name, eq) in enumerate(zip(args[1:], exprs, strict=True)):
            eqs.append(f"[{i:02d}]:\n\td{stringify(name)}/dt = {stringify(eq)}")

        return "\n".join(eqs)


# }}}
