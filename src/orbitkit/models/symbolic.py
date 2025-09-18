# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, Protocol, TypeVar

import numpy as np
import sympy as sp

from orbitkit.typing import Array, DataclassInstanceT
from orbitkit.utils import module_logger

log = module_logger(__name__)


T = TypeVar("T")

# {{{ symbolic

Variable = int | float | sp.Expr


def make_variable(name: str) -> sp.Symbol:
    return sp.Symbol(name, real=True)


var = make_variable


def make_sym_vector(name: str, dim: int) -> sp.Symbol | Array:
    if dim == 0:
        return make_variable(name)

    result = np.empty((dim,), dtype=object)

    x = sp.IndexedBase(name, shape=(dim,), real=True)
    for i in range(dim):
        result[i] = x[i]

    return result


def make_sym_function(name: str, dim: int) -> Array:
    # NOTE: this corresponds to how jitcode works, so we have a little helper
    result = np.empty((dim,), dtype=object)
    for i in range(dim):
        result[i] = sp.Function(name)(i)

    return result


def ds_symbolic(
    obj: DataclassInstanceT,
    *,
    rec: bool = False,
    rattrs: set[str] | None = None,
) -> DataclassInstanceT:
    """Fill in all the attributes of *cls* with symbolic variables.

    :arg rec: if *True*, automatically recurse into all child dataclasses.
    :arg attrs: a set of attribute names that will be recursed into regardless
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

        if callable(attr):
            kwargs[f.name] = sp.Function(f.name)
        elif isinstance(attr, tuple):
            kwargs[f.name] = tuple(
                sp.Function(f"{f.name}_{i}")
                if callable(attr[i])
                else sp.Symbol(f"{f.name}_{i}", real=True)
                for i in range(len(attr))
            )
        else:
            kwargs[f.name] = sp.Symbol(f.name, real=True)

    return replace(obj, **kwargs)


def lambdify(
    exprs: Array,
    *args: sp.Symbol,
    modules: str = "numpy",
) -> Callable[[float, Array], Array]:
    """A wrapper around :func:`~sympy.utilities.lambdify.lambdify` that works
    for the models.

    This creates a callable wrapper that takes :math:`(t, y)` as inputs and returns
    an array of the same size as :math:`y`. This is meant to be used with
    integrators such as those from :mod:`scipy`.
    """

    func = sp.lambdify(args, exprs, modules=modules)
    nargs = len(args)

    def wrapper(t: float, y: Array) -> Array:
        d = nargs - 1
        if y.size % d != 0:
            raise ValueError("inputs do not match required arguments")

        # get size of each variable
        n = y.size // d

        # make sure all the entries are that size
        ts = np.full((n,), t, dtype=y.dtype)
        ys = np.array_split(y, d)

        # evaluate
        return func(ts, *ys)  # type: ignore[no-any-return]

    return wrapper


# }}}


# {{{ models


@dataclass(frozen=True)
class Model(ABC):
    @property
    @abstractmethod
    def variables(self) -> tuple[str, ...]:
        """
        :returns: a tuple of all the state variables in the system.
        """

    @abstractmethod
    def evaluate(self, t: float, *args: Array) -> Array:
        """
        :returns: an expression of the model evaluated at the given arguments.
        """

    def lambdify(self, n: int) -> Callable[[float, Array], Array]:
        """Create a callable that is usable by :func:`scipy.integrate.solve_ivp`
        or other similar integrators.

        This uses :meth:`variables` and :class:`lambdify` to create a
        :mod:`numpy` compatible callable.
        """
        t = sp.Symbol("t")
        args = [make_sym_vector(name, n) for name in self.variables]
        expr = self.evaluate(t, *args)

        return lambdify(expr, t, *args)

    def symbolic(self, *, rec: bool = False) -> Array:
        """Create a completely symbolic version of this model.

        All the parameters in the model will be replaced with symbolic variables.
        These will then be evaluated into a symbolic expression. The expression
        will usually be a scalar expression for each state variables.
        """
        t = sp.Symbol("t")
        args = [make_sym_vector(name, 0) for name in self.variables]

        model = ds_symbolic(self, rec=rec, rattrs={"param"})
        return model.evaluate(t, *args)

    def pretty(self, *, use_unicode: bool = True) -> tuple[str, ...]:
        result = []
        for name, expr in zip(self.variables, self.symbolic(), strict=True):
            t = sp.Symbol("t")
            dy = sp.Derivative(sp.Function(name)(t), t)
            result.append(sp.pretty(sp.Eq(dy, expr), use_unicode=use_unicode))

        return tuple(result)


# }}}

# {{{ rate functions


def vectorize(func: Callable[[T], T], x: T) -> T:
    if isinstance(x, np.ndarray):
        return np.vectorize(func)(x)  # type: ignore[no-any-return]
    else:
        return func(x)


class RateFunction(Protocol):
    """A callable protocol for rate functions used in many neuron models.

    .. automethod:: __call__
    """

    def __call__(self, V: Array) -> Array:
        """Evaluate the rate function for the given membrane potential."""


@dataclass(frozen=True)
class ExponentialRate:
    r"""Exponential rate function.

    .. math::

        f(V; A, \theta, \sigma) = A \exp\left(-\frac{(V - \theta)}{\sigma}\right).
    """

    a: float
    theta: float
    sigma: float

    def __call__(self, V: Array) -> Array:
        return self.a * vectorize(sp.exp, -(V - self.theta) / self.sigma)


@dataclass(frozen=True)
class SigmoidRate:
    r"""Sigmoid rate function.

    .. math::

        f(V; A, \theta, \sigma) =
            \frac{A}{1 + \exp\left(-\frac{(V - \theta)}{\sigma}\right)}.
    """

    a: float
    theta: float
    sigma: float

    def __call__(self, V: Array) -> Array:
        expV = vectorize(sp.exp, -(V - self.theta) / self.sigma)
        return self.a / (1.0 + expV)


@dataclass(frozen=True)
class Expm1Rate:
    r"""Reciprocal of the ``expm1`` function.

    .. math::

        f(V; A, \theta, \sigma) =
            \frac{A}{1 - \exp\left(-\frac{(V - \theta)}{\sigma}\right)}.
    """

    a: float
    theta: float
    sigma: float

    def __call__(self, V: Array) -> Array:
        expV = vectorize(sp.exp, -(V - self.theta) / self.sigma)
        return self.a / (1.0 - expV)


@dataclass(frozen=True)
class LinearExpm1Rate:
    r"""A linear exponential rate function.

    .. math::

        f(V; a, b, \theta, \sigma) =
            \frac{a V + b}{1 - \exp\left(-\frac{(V - \theta)}{\sigma}\right)}
    """

    a: float
    b: float
    theta: float
    sigma: float

    def __call__(self, V: Array) -> Array:
        expV = vectorize(sp.exp, -(V - self.theta) / self.sigma)
        return (self.a * V + self.b) / (1.0 - expV)


@dataclass(frozen=True)
class TanhRate:
    r"""A :math:`tanh` based rate function.

    .. math::

        f(V; A, theta, sigma) =
            A \left[1 + \tanh\left(\frac{V - \theta}{\sigma}\right)\right].
    """

    a: float
    theta: float
    sigma: float

    def __call__(self, V: Array) -> Array:
        tanhV = vectorize(sp.tanh, (V - self.theta) / self.sigma)
        return self.a * (1.0 + tanhV)


# }}}
