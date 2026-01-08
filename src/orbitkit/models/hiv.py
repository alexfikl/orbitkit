# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import orbitkit.symbolic.primitives as sym
from orbitkit.models import Model
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ CulshawRuanWebb


@dataclass(frozen=True)
class CulshawRuanWebb(Model):
    r"""Right-hand side of the HIV model from [Culshaw2003]_.

    .. math::

        \begin{aligned}
        \frac{\mathrm{d} C}{\mathrm{d} t} & =
            (r - \mu_C) C \left(1 - \frac{C + I}{C_M}\right) - k_I I C, \\
        \frac{\mathrm{d} I}{\mathrm{d} t} & =
            k_I' (h \ast (I C)) - \mu_I I,
        \end{aligned}

    where :math:`h * (I C)` is the convolution with the delay kernel :math:`h`.
    The state variables :math:`C` and :math:`I` represent the concentration of
    healthy and infected cells, respectively. The fraction :math:`f = k_I' / k_I`
    gives the fraction of cells that survive the incubation period. The remaining
    parameters and their respective units are given in Table 1 from [Culshaw2003]_.

    .. [Culshaw2003] R. V. Culshaw, S. Ruan, G. Webb,
        *A Mathematical Model of Cell-to-Cell Spread of HIV-1 That Includes a
        Time Delay*,
        Journal of Mathematical Biology, Vol. 46, pp. 425--444, 2003,
        `doi:10.1007/s00285-002-0191-5 <https://doi.org/10.1007/s00285-002-0191-5>`__.
    """

    r: sym.Expression
    """Healthy cell reproductive rate."""
    C_M: sym.Expression
    """Effective carrying capacity of healthy cells."""
    k_I: sym.Expression
    """Rate constant for cell-to-cell spread."""
    f: sym.Expression
    """Fraction of cells that survive the incubation period."""
    mu_C: sym.Expression
    """Death rate of healthy cells."""
    mu_I: sym.Expression
    """Death rate of infected cells."""

    h: sym.DelayKernel
    """Delay kernel used in the second equation."""

    @property
    def n(self) -> int:
        return 1

    @property
    def variables(self) -> tuple[str, ...]:
        return ("C", "I")

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        C, I = args  # noqa: E741

        r, C_M, k_I, mu_C, mu_I = self.r, self.C_M, self.k_I, self.mu_C, self.mu_I
        r_C = r - mu_C
        k_Ip = k_I * self.f

        return (
            r_C * C * (1.0 - (C + I) / C_M) - k_I * C * I,
            k_Ip * self.h(I * C) - mu_I * I,
        )


# }}}


# {{{


def _make_hiv_culshaw_2003(kernel: sym.DelayKernel, k_Ip: float) -> CulshawRuanWebb:
    k_I = 2.0 * 1.0e-6
    f = k_Ip / k_I

    return CulshawRuanWebb(
        r=0.7,
        C_M=2.0 * 1.0e6,
        k_I=k_I,
        f=f,
        mu_C=0.02,
        mu_I=0.3,
        h=kernel,
    )


HIV_MODEL = {
    "CulshawRuanWebb2003Figure32": _make_hiv_culshaw_2003(
        sym.DiracDelayKernel(0.4), 1.5 * 1.0e-7
    ),
    "CulshawRuanWebb2003Figure42": _make_hiv_culshaw_2003(
        sym.DiracDelayKernel(0.4), 1.0e-6
    ),
    "CulshawRuanWebb2003Figure44": _make_hiv_culshaw_2003(
        sym.DiracDelayKernel(1.0), 1.0e-6
    ),
    "CulshawRuanWebb2003Figure52": _make_hiv_culshaw_2003(
        sym.GammaDelayKernel(1, 5.0), 1.5 * 1.0e-6
    ),
    "CulshawRuanWebb2003Figure54": _make_hiv_culshaw_2003(
        sym.GammaDelayKernel(1, 1.5), 1.5 * 1.0e-6
    ),
}


def get_registered_parameters() -> tuple[str, ...]:
    return tuple(HIV_MODEL)


def make_model_from_name(name: str) -> Model:
    return HIV_MODEL[name]


# }}}
