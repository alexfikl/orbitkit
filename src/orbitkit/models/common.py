# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np
import symengine as sp

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
    """A generic rate function used in many neuron models."""

    def __call__(self, V: Array) -> Array: ...


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
            \frac{A}{1 + \exp\left(-\frac{V - \theta}{\sigma}\right)}.
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
            \frac{A}{1 - \exp\left(-\frac{V - \theta}{\sigma}\right)}.
    """

    amplitude: float
    theta: float
    sigma: float

    def __call__(self, V: Array) -> Array:
        expV = vectorize(sp.exp, -(V - self.theta) / self.sigma)
        return self.amplitude / (1.0 - expV)


def make_sym_vector(name: str, dim: int) -> Array:
    result = np.empty((dim,), dtype=object)
    for i in range(dim):
        # TODO: ideally this would use something like sympy.IndexedBase, but
        # symengine does not have that yet. Not clear if this works well enough..
        result[i] = sp.Symbol(f"{name}{i}")

    return result


def make_sym_function(name: str, dim: int) -> Array:
    # NOTE: this corresponds to how jitcode works, so we have a little helper
    result = np.empty((dim,), dtype=object)
    for i in range(dim):
        result[i] = sp.Function(name)(i)

    return result
