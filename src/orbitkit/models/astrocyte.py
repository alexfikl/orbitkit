# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import orbitkit.symbolic.primitives as sym
from orbitkit.models import Model
from orbitkit.models.rate_functions import LinearRationalRate, RateFunction
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ LiRinzel model


@dataclass(frozen=True)
class LiRinzelParameter:
    """Parameters for the astrocyte calcium model from [LiRinzel1994]."""

    InsP3: float
    """:math:`\text{InsP}_3` concentration (microM)."""

    c_0: float
    """Total free calcium per cytosolic volume (microM)."""
    c_1: float
    """Volume ratio between the endoplasmic reticulum and cytosol."""

    v_1: float
    """Maximum :math:`\text{InsP}_3 R` permeability (Hz)."""
    v_2: float
    """Maximum leak permeability (Hz)."""
    v_3: float
    """Maximum pump rate (Hz)."""
    k_3: float
    r"""The dissociation constant of :math:`\text{Ca}^{2+}` to the pump (microM)."""

    a_2: float
    """The on-rate for :math:`\text{Ca}^{2+}` binding to the inactivation site
    (Hz/microM)."""


@dataclass(frozen=True)
class LiRinzel(Model):
    r"""Right-hand side of an model from Equation (6) in [LiRinzel1994]_.

    .. math::

        \begin{cases}
        \frac{\mathrm{d} C}{\mathrm{d} t} & =
            c_1 v_1 m_\infty^3(I) n_\infty^3(C) h^3 (C_{\text{ER}} - C)
            + c_1 v_2 (C_{\text{ER}} - C)
            - v_3 \frac{C^2}{C^2 + k_3^2}, \\
        \frac{\mathrm{d} h}{\mathrm{d} t} & =
            \frac{h_\infty(C, I) - h_i}{\tau_h(C, I)},
        \end{cases}

    where :math:`C` is the cytosolic :math:`\text{Ca}^{2+}` concentration
    and :math:`h` is a slow inactivation variable. The additional variables
    are given by

    .. math::

        C_{\text{ER}}(C) = \frac{c_0 - C}{c_1},
        I = \text{InsP}_3.

    We note that, following [LiRinzel1994]_, :math:`C` has units of micromolars
    and :math:`h` is dimensionless. The remaining parameters are documented in
    :class:`LiRinzelParameter`.

    .. [LiRinzel1994] Y.-X. Li, J. Rinzel,
        *Equations for :math:`\text{InsP}_3` Receptor-Mediated
        :math:`[\text{Ca}^{2+}]_i` Oscillations Derived From a Detailed
        Kinetic Model: A Hodgkin-Huxley Like Formalism*,
        Journal of Theoretical Biology, Vol. 166, pp. 461--473, 1994,
        `doi:10.1006/jtbi.1994.1041 <https://doi.org/10.1006/jtbi.1994.1041>`__.
    """

    param: LiRinzelParameter
    """Parameters in the Li-Rinzel model."""

    minf: RateFunction
    r"""Steady-state activation function :math:`m_\infty(I)`."""
    ninf: RateFunction
    r"""Steady-state activation function :math:`n_\infty(C)`."""
    Q2: RateFunction
    """Rate function for the inactivation variable :math:`h`."""

    @property
    def variables(self) -> tuple[str, ...]:
        return ("C", "h")

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        C, h = args
        param = self.param

        # NOTE: Equations (6) and (7) in [LiRinzel1994].
        C_ER = (param.c_0 - C) / param.c_1
        V_1 = param.v_1 * param.c_1
        V_2 = param.v_2 * param.c_1
        V_3 = param.v_3
        IP3 = param.InsP3

        minf = self.minf(IP3) * self.ninf(C)

        Q2 = self.Q2(IP3)
        hinf = Q2 / (Q2 + C)
        tau = 1 / (param.a_2 * (Q2 + C))

        return (
            V_1 * minf**3 * h**3 * (C_ER - C)
            + V_2 * (C_ER - C)
            - V_3 * C**2 / (param.k_3**2 + C**2),
            (hinf - h) / tau,
        )


# }}}


@dataclass(frozen=True)
class DePitta(LiRinzel):
    """
    .. [DePitta2009] M. De Pittà, M. Goldberg, V. Volman, H. Berry, E. Ben-Jacob,
        *Glutamate Regulation of Calcium and IP3 Oscillating and Pulsating
        Dynamics in Astrocytes*,
        Journal of Biological Physics, Vol. 35, pp. 383--411, 2009,
        `doi:10.1007/s10867-009-9155-y <https://doi.org/10.1007/s10867-009-9155-y>`__.
    """


# {{{ parameters


def _make_li_rinzel_1994(InsP3: float) -> LiRinzel:
    # NOTE: See Table 1 in [LiRinzel1994]
    k_3 = 0.1
    c_1 = 0.185

    a_1 = 40.0 / k_3
    a_2 = 0.02 / k_3
    a_3 = 40.0 / k_3
    a_5 = 2.0 / k_3

    b_1 = 52.0
    b_2 = 0.2098
    b_3 = 377.36
    b_5 = 1.6468

    param = LiRinzelParameter(
        InsP3=InsP3,
        c_0=2.0,
        c_1=c_1,
        v_1=7.11 / (1 + c_1),
        v_2=0.13035 / (1 + c_1),
        v_3=9.0 * k_3,
        k_3=k_3,
        a_2=a_2,
    )

    return LiRinzel(
        param=param,
        minf=LinearRationalRate(1.0, 0.0, b_1 / a_1),
        ninf=LinearRationalRate(1.0, 0.0, b_5 / a_5),
        Q2=LinearRationalRate(b_2 / a_2, b_1 / a_1, b_3 / a_3),
    )


ASTROCYTE_MODEL = {
    "LiRinzel1994Figure3": _make_li_rinzel_1994(0.4),
}


def get_registered_parameters() -> tuple[str, ...]:
    return tuple(ASTROCYTE_MODEL)


def make_model_from_name(name: str) -> Model:
    return ASTROCYTE_MODEL[name]


# }}}
