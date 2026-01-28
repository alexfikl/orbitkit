# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import orbitkit.symbolic.primitives as sym
from orbitkit.models import Model
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ MackeyGlass


@dataclass(frozen=True)
class MackeyGlass1(Model):
    r"""Right-hand side of the first Mackey-Glass model from [Mackey1977]_.

    .. math::

        \frac{\mathrm{d} P}{\mathrm{d} t} =
            \frac{\beta \theta^k}{\theta^k + (h \ast P)^k} - \gamma P,

    where :math:`h \ast P` is the convolution of the variable with a distributed
    delay kernel.

    .. [Mackey1977] M. C. Mackey, L. Glass,
        *Oscillation and Chaos in Physiological Control Systems*,
        Science, Vol. 197, pp. 287--289, 1977,
        `doi:10.1126/science.267326 <https://doi.org/10.1126/science.267326>`__.
    """

    beta: sym.Expression
    """Production rate in the Mackey-Glass model."""
    gamma: sym.Expression
    """Decay constant in the Mackey-Glass model."""
    theta: sym.Expression
    """(Half-)Saturation point in the Mackey-Glass model. This constant usually
    disappears under normalization.
    """
    k: sym.Expression
    """Parameter that controls the steepness of the feedback (see Hill coefficient)."""

    h: sym.DelayKernel
    """Delay kernel used in the equation."""

    @property
    def n(self) -> int:
        return 1

    @property
    def variables(self) -> tuple[str, ...]:
        return ("P",)

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        (P,) = args

        beta, gamma, theta, k = self.beta, self.gamma, self.theta, self.k
        return (beta * theta**k / (theta**k + self.h(P) ** k) - gamma * P,)


@dataclass(frozen=True)
class MackeyGlass2(MackeyGlass1):
    r"""Right-hand side of the second Mackey-Glass model from [Mackey1977]_.

    .. math::

        \frac{\mathrm{d} P}{\mathrm{d} t} =
            \frac{\beta_0 \theta^k (h \ast P)^k}{\theta^k + (h \ast P)^k} - \gamma P.
    """

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        (P,) = args

        beta, gamma, theta, k = self.beta, self.gamma, self.theta, self.k
        return (
            beta * theta**k * self.h(P) ** k / (theta**k + self.h(P) ** k) - gamma * P,
        )


# }}}

# {{{ parameters


def _make_mackey_glass_1977_figure2(tau: float) -> Model:
    return MackeyGlass2(
        beta=0.2, gamma=0.1, k=10, theta=1.0, h=sym.DiracDelayKernel(tau)
    )


def _make_mackey_glass_1979_figure6(k: float) -> Model:
    return MackeyGlass2(
        beta=2.0,
        gamma=1.0,
        k=k,
        theta=1.0,
        h=sym.DiracDelayKernel(2.0),
    )


MACKEY_GLASS_MODEL = {
    "MackeyGlass1977Figure2b": _make_mackey_glass_1977_figure2(6.0),
    "MackeyGlass1977Figure2c": _make_mackey_glass_1977_figure2(20.0),
    "MackeyGlass1979Figure6a": _make_mackey_glass_1979_figure6(7.0),
    "MackeyGlass1979Figure6b": _make_mackey_glass_1979_figure6(7.75),
    "MackeyGlass1979Figure6c": _make_mackey_glass_1979_figure6(8.50),
    "MackeyGlass1979Figure6d": _make_mackey_glass_1979_figure6(8.79),
    "MackeyGlass1979Figure6e": _make_mackey_glass_1979_figure6(9.65),
    "MackeyGlass1979Figure6f": _make_mackey_glass_1979_figure6(9.69715),
    "MackeyGlass1979Figure6g": _make_mackey_glass_1979_figure6(9.6975),
    "MackeyGlass1979Figure6h": _make_mackey_glass_1979_figure6(9.76),
    "MackeyGlass1979Figure6i": _make_mackey_glass_1979_figure6(10.0),
    "MackeyGlass1979Figure6j": _make_mackey_glass_1979_figure6(20.0),
}


def get_registered_parameters() -> tuple[str, ...]:
    return tuple(MACKEY_GLASS_MODEL)


def make_model_from_name(name: str) -> Model:
    return MACKEY_GLASS_MODEL[name]


# }}}
