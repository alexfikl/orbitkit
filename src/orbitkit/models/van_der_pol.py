# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import orbitkit.symbolic.primitives as sym
from orbitkit.models import Model
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ VanDerPol


@dataclass(frozen=True)
class VanDerPol(Model):
    r"""Right-hand side of the van der Pol oscillator.

    .. math::

        \begin{aligned}
        \frac{\mathrm{d} x}{\mathrm{d} t} & =
            y, \\
        \frac{\mathrm{d} y}{\mathrm{d} t} & =
            \mu (1 - x^2) y - x + A \sin (\omega t).
        \end{aligned}
    """

    mu: float
    """Parameter indicating the strength of the nonlinearity and damping."""
    amplitude: float
    """Forcing amplitude."""
    omega: float
    """Angular velocity of the forcing."""

    @property
    def n(self) -> int:
        return 1

    @property
    def variables(self) -> tuple[str, ...]:
        return ("x", "y")

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        x, y = args

        f = self.amplitude * sym.cos(self.omega * t)
        return (y, self.mu * (1.0 - x**2) * y - x + f)


# }}}
