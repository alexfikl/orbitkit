# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Protocol, TypeAlias

import numpy as np
import pymbolic.primitives as prim
from pymbolic.mapper.stringifier import PREC_NONE
from pymbolic.mapper.stringifier import StringifyMapper as StringifyMapperBase

from orbitkit.typing import Array, DataclassInstanceT
from orbitkit.utils import module_logger

log = module_logger(__name__)

Expression: TypeAlias = (
    int | float | complex | np.inexact | np.integer | prim.ExpressionNode
)


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
    def make_stringifier(  # noqa: PLR6301
        self,
        originating_stringifier: StringifyMapperBase[Any] | None = None,
    ) -> StringifyMapper:
        return StringifyMapper()


@prim.expr_dataclass()
class Contract(ExpressionNode):
    aggregate: Expression
    axis: tuple[int, ...]


@prim.expr_dataclass()
class DotProduct(ExpressionNode):
    left: Array | Expression
    right: Array | Expression


@prim.expr_dataclass()
class Reshape(ExpressionNode):
    aggregate: Expression
    shape: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)


@prim.expr_dataclass()
class MatrixSymbol(prim.Variable):
    shape: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def reshape(self, *shape: int) -> Reshape:
        return Reshape(self, shape)


# }}}


# {{{ functions


@prim.expr_dataclass()
class Function(prim.Variable):
    pass


sin = Function("sin")
cos = Function("cos")
exp = Function("exp")
tanh = Function("tanh")

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


# {{{ stringifier


class StringifyMapper(StringifyMapperBase[Any]):
    def map_variable(self, expr: prim.Variable, enclosing_prec: int) -> str:  # noqa: PLR6301
        from sympy.printing.pretty.pretty_symbology import pretty_symbol

        return str(pretty_symbol(expr.name))

    def map_numpy_array(  # noqa: PLR6301
        self, expr: np.ndarray[tuple[int, ...], np.dtype[Any]], enclosing_prec: int
    ) -> str:
        return repr(expr)

    def map_contract(self, expr: Contract, enclosing_prec: int) -> str:
        aggregate = self.rec(expr.aggregate, PREC_NONE)
        return f"sum({aggregate}, axis={expr.axis})"

    def map_reshape(self, expr: Reshape, enclosing_prec: int) -> str:
        aggregate = self.rec(expr.aggregate, PREC_NONE)
        return f"({aggregate}).reshape{expr.shape}"

    def map_dot_product(self, expr: DotProduct, enclosing_prec: int) -> str:
        left = self.rec(expr.left, PREC_NONE)  # type: ignore[arg-type]
        right = self.rec(expr.right, PREC_NONE)  # type: ignore[arg-type]
        return f"dot({left}, {right})"


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
            variables (i.e. :class:`~sympy.Symbol`\ s and such) and the model is
            given as a symbolic object array.
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


# }}}
