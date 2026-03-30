# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import orbitkit.symbolic.primitives as sym
from orbitkit.models import Model
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ ThetaModel


@dataclass(frozen=True)
class ThetaModel(Model):
    r"""Right-hand side of the Theta neuron model from Equation 21 in [Laing2018]_.

    .. math::

        \frac{\mathrm{d} z}{\mathrm{d} t} =
            i (\eta + \kappa (h \ast I) + 1) z
            + i(\eta + \kappa (h \ast I) - 1) \frac{1 + z^2}{2},

    where :math:`z` is a complex variable. This implementation converts this to
    a two component model for the real and imaginary parts instead. The
    :math:`h \ast I` notation denotes the convolution of the distributed delay
    kernel :math:`h` with the synaptic current :math:`I` given by

    .. math::

        I = \frac{3}{2} - (z + \bar{z}) + \frac{z^2 + \bar{z}^2}{4}.

    .. [Laing2018] C. R. Laing,
        *The Dynamics of Networks of Identical Theta Neurons*,
        The Journal of Mathematical Neuroscience, Vol. 8, pp. 4--4, 2018,
        `doi:10.1186/s13408-018-0059-7 <https://doi.org/10.1186/s13408-018-0059-7>`__.
    """

    eta: sym.Expression
    """Input current to all the neurons when uncoupled."""
    kappa: sym.Expression
    """Coupling strength."""

    h: sym.DelayKernel
    """Delay in the synaptic current."""

    @property
    def n(self) -> int:
        return 1

    @property
    def variables(self) -> tuple[str, ...]:
        return ("z_re", "z_im")

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        z_re, z_im = args

        eta = self.eta
        kappa = self.kappa
        h = self.h

        Ire = 1.5 - 2 * h(z_re) + (h(z_re) ** 2 - h(z_im) ** 2) / 2
        cp = eta + kappa * Ire + 1
        cm = eta + kappa * Ire - 1

        return (
            -cp * z_im - cm * z_re * z_im,
            cp * z_re + cm * (1 + z_re**2 - z_im**2) / 2,
        )


# }}}


# {{{ make_model_from_name


def _make_laing_2018(eta: float, kappa: float) -> ThetaModel:
    return ThetaModel(
        eta=eta,
        kappa=kappa,
        h=sym.ZeroDelayKernel(),
    )


THETA_MODEL = {
    "Laing2018Figure2a": _make_laing_2018(-0.5, 1.0),
    "Laing2018Figure2b": _make_laing_2018(+0.5, 1.0),
}


def get_registered_parameters() -> tuple[str, ...]:
    return tuple(THETA_MODEL)


def make_model_from_name(name: str) -> ThetaModel:
    return THETA_MODEL[name]


# }}}
