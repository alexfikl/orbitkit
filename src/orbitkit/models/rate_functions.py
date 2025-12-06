# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import orbitkit.symbolic.primitives as sym


class RateFunction(Protocol):
    """A callable protocol for rate functions used in many neuron models.

    .. automethod:: __call__
    """

    def __call__(self, V: sym.Expression) -> sym.Expression:
        """Evaluate the rate function for the given membrane potential."""


@dataclass(frozen=True)
class ExponentialRate:
    r"""Exponential rate function.

    .. math::

        f(V; a, \theta, \sigma) = a \exp\left(-\frac{(V - \theta)}{\sigma}\right).
    """

    a: sym.Expression
    theta: sym.Expression
    sigma: sym.Expression

    def __call__(self, V: sym.Expression) -> sym.Expression:
        return self.a * sym.exp(-(V - self.theta) / self.sigma)


@dataclass(frozen=True)
class SigmoidRate:
    r"""Sigmoid rate function.

    .. math::

        f(V; a, \theta, \sigma) =
            \frac{a}{1 + \exp\left(-\frac{(V - \theta)}{\sigma}\right)}.
    """

    a: sym.Expression
    theta: sym.Expression
    sigma: sym.Expression

    def __call__(self, V: sym.Expression) -> sym.Expression:
        return self.a / (1.0 + sym.exp(-(V - self.theta) / self.sigma))


@dataclass(frozen=True)
class Expm1Rate:
    r"""Reciprocal of the ``expm1`` function.

    .. math::

        f(V; a, \theta, \sigma) =
            \frac{a}{1 - \exp\left(-\frac{(V - \theta)}{\sigma}\right)}.
    """

    a: sym.Expression
    theta: sym.Expression
    sigma: sym.Expression

    def __call__(self, V: sym.Expression) -> sym.Expression:
        return self.a / (1.0 - sym.exp(-(V - self.theta) / self.sigma))


@dataclass(frozen=True)
class LinearExpm1Rate:
    r"""A linear exponential rate function.

    .. math::

        f(V; a, b, \theta, \sigma) =
            \frac{a V + b}{1 - \exp\left(-\frac{(V - \theta)}{\sigma}\right)}
    """

    a: sym.Expression
    b: sym.Expression
    theta: sym.Expression
    sigma: sym.Expression

    def __call__(self, V: sym.Expression) -> sym.Expression:
        return (self.a * V + self.b) / (1.0 - sym.exp(-(V - self.theta) / self.sigma))


@dataclass(frozen=True)
class TanhRate:
    r"""A :math:`tanh` based rate function.

    .. math::

        f(V; A, \theta, \sigma) =
            A \left[1 + \tanh\left(\frac{V - \theta}{\sigma}\right)\right].
    """

    a: sym.Expression
    theta: sym.Expression
    sigma: sym.Expression

    def __call__(self, V: sym.Expression) -> sym.Expression:
        return self.a * (1.0 + sym.tanh((V - self.theta) / self.sigma))
