# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np
import sympy as sp

from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


T = TypeVar("T")


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

    amplitude: float
    theta: float
    sigma: float

    def __call__(self, V: Array) -> Array:
        return self.amplitude * vectorize(sp.exp, -(V - self.theta) / self.sigma)


@dataclass(frozen=True)
class SigmoidRate:
    r"""Sigmoid rate function.

    .. math::

        f(V; A, \theta, \sigma) =
            \frac{A}{1 + \exp\left(-\frac{(V - \theta)}{\sigma}\right)}.
    """

    amplitude: float
    theta: float
    sigma: float

    def __call__(self, V: Array) -> Array:
        expV = vectorize(sp.exp, -(V - self.theta) / self.sigma)
        return self.amplitude / (1.0 + expV)


@dataclass(frozen=True)
class Expm1Rate:
    r"""Reciprocal of the ``expm1`` function.

    .. math::

        f(V; A, \theta, \sigma) =
            \frac{A}{1 - \exp\left(-\frac{(V - \theta)}{\sigma}\right)}.
    """

    amplitude: float
    theta: float
    sigma: float

    def __call__(self, V: Array) -> Array:
        expV = vectorize(sp.exp, -(V - self.theta) / self.sigma)
        return self.amplitude / (1.0 - expV)


def make_variable(name: str) -> sp.Symbol:
    return sp.Symbol(name, real=True)


var = make_variable


def make_sym_vector(name: str, dim: int) -> Array:
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


class lambdify:  # noqa: N801
    """A wrapper around :func:`sympy.lambdify` that works for the models.

    This creates a callable wrapper that takes :math:`(t, y)` as inputs and returns
    an array of the same size as :math:`y`. This is meant to be used with
    integrators such as those from :mod:`scipy`.
    """

    exprs: Array
    args: tuple[sp.Symbol, ...]
    func: Callable[..., Array]

    def __init__(
        self,
        exprs: Array,
        *args: sp.Symbol,
        backend: str = "lambda",
    ) -> None:
        self.exprs = exprs
        self.args = args

        self.func = sp.lambdify(
            (sp.Symbol("t"), *args),
            exprs,
            modules="numpy",
        )

    @property
    def nargs(self) -> int:
        return len(self.args)

    def __call__(self, t: float, y: Array) -> Array:
        if y.size % self.nargs != 0:
            raise ValueError("inputs do not match required arguments")

        # get size of each variable
        n = y.size // self.nargs

        # make sure all the entries are that size
        ts = np.full((n,), t, dtype=y.dtype)
        ys = np.array_split(y, self.nargs)
        assert all(y.shape == (n,) for y in ys)

        # evaluate
        return self.func(ts, *ys)
