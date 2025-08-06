# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np

from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


class RateFunction(Protocol):
    """A generic rate function used in many neuron models."""

    def __call__(self, V: Array) -> Array: ...


class ExponentialRate(NamedTuple):
    r"""Exponential rate function.

    .. math::

        f(V; A, \theta, \sigma) = A \exp\left(-\frac{(V - \theta)}{\sigma}\right)

    """

    amplitude: float
    theta: float
    sigma: float

    def __call__(self, V: Array) -> Array:
        return self.amplitude * np.exp(-(V - self.theta) / self.sigma)


class SigmoidRate(NamedTuple):
    r"""Sigmoid rate function.

    .. math::

        f(V; A, \theta, \sigma) =
            \frac{A}{1 + \exp\left(-\frac{V - \theta}{\sigma}\right)}.
    """

    amplitude: float
    theta: float
    sigma: float

    def __call__(self, V: Array) -> Array:
        expV = np.exp(-(V - self.theta) / self.sigma)
        return self.amplitude / (1.0 + expV)


class Expm1Rate(NamedTuple):
    r"""Reciprocal of the ``expm1`` function.

    .. math::

        f(V; A, \theta, \sigma) =
            \frac{A}{1 - \exp\left(-\frac{V - \theta}{\sigma}\right)}.
    """

    amplitude: float
    theta: float
    sigma: float

    def __call__(self, V: Array) -> Array:
        expV = np.exp(-(V - self.theta) / self.sigma)
        return self.amplitude / (1.0 - expV)
