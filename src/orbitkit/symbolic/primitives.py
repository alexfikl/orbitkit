# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import pymbolic.primitives as prim
from pymbolic.mapper.stringifier import StringifyMapper as StringifyMapperBase

from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)

Expression: TypeAlias = (
    int | float | complex | np.inexact | np.integer | prim.ExpressionNode
)
"""A union of allowable classes that can be part of an expression."""
ArrayExpression: TypeAlias = Expression | np.ndarray[tuple[int, ...], np.dtype[Any]]
"""An expression that can also be an array of expressions."""


# {{{ expressions


Variable = prim.Variable


@prim.expr_dataclass()
class ExpressionNode(prim.ExpressionNode):
    """A base class for ``orbitkit``-specific expression nodes."""

    def make_stringifier(  # noqa: PLR6301
        self,
        originating_stringifier: StringifyMapperBase[Any] | None = None,
    ) -> StringifyMapperBase[Any]:
        from orbitkit.symbolic.mappers import StringifyMapper

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
class VariableWithDelay(Variable):
    r"""A variable with a constant delay, e.g. :math:`y(t - \tau)`."""

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
class DiracDelayKernel(DelayKernel):
    r"""A delay kernel based on the Dirac distribution.

    .. math::

        \mathrm{Dirac}(t; \tau) = \delta(t - \tau).
    """

    tau: Expression
    """Average delay of the kernel."""


@prim.expr_dataclass()
class GammaDelayKernel(DelayKernel):
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
class UniformDelayKernel(DelayKernel):
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
class TriangularDelayKernel(DelayKernel):
    r"""A delay kernel based on the triangular distribution.

    .. math::

        \mathrm{Triangular}(t; \epsilon, \tau) =
        \begin{cases}
        \dfrac{t - (1 - \epsilon) \tau}{(\epsilon \tau)^2}, &
            \quad (1 - \epsilon) \tau < t < \tau, \\
        \dfrac{(1 + \epsilon) \tau - t}{(\epsilon \tau)^2}, &
            \quad \tau \le t < (1 + \epsilon) \tau, \\
        0, & \quad \text{otherwise}.
        \end{cases}
    """

    epsilon: Expression
    """Parameter controlling the width of the base of the triangle."""
    tau: Expression
    """Average delay of the kernel."""


# }}}
