# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import orbitkit.symbolic.primitives as sym
from orbitkit.models import Model
from orbitkit.models.rate_functions import HillRate, LinearRationalRate, RateFunction
from orbitkit.utils import module_logger

if TYPE_CHECKING:
    from collections.abc import Callable

log = module_logger(__name__)


# {{{ LiRinzel model


@dataclass(frozen=True)
class LiRinzelParameter:
    """Parameters for the astrocyte calcium model from [LiRinzel1994]_."""

    # [Ca] equation
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

    # [h] equation
    a_2: float
    """The on-rate for :math:`\text{Ca}^{2+}` binding to the inactivation site
    (Hz)."""


@dataclass(frozen=True)
class LiRinzel(Model):
    r"""Right-hand side of an model from Equation (6) in [LiRinzel1994]_.

    .. math::

        \begin{aligned}
        \frac{\mathrm{d} C}{\mathrm{d} t} & =
            c_1 v_1 m_\infty^3(I) n_\infty^3(C) h^3 (C_{\text{ER}} - C)
            + c_1 v_2 (C_{\text{ER}} - C)
            - v_3 \frac{C^2}{C^2 + k_3^2}, \\
        \frac{\mathrm{d} h}{\mathrm{d} t} & =
            \frac{h_\infty(C, I) - h_i}{\tau_h(C, I)},
        \end{aligned}

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
            - V_3 * sym.hill2(C, param.k_3),
            (hinf - h) / tau,
        )


# }}}


# {{{{ DePitta model


@dataclass(frozen=True)
class DePittaParameter:
    r"""Parameters used in the De Pittà model described in Table 1 from [DePitta2009]_.

    These parameters are mainly an extension of :class:`LiRinzelParameter` with
    different names. In particular, :math:`r_C \equiv v_1`, :math:`r_L \equiv v_2`,
    :math:`v_{ER} \equiv v_3` and :math:`k_{ER} \equiv k_3`.
    """

    # [Ca] equation
    c_0: float
    """Total free calcium per cytosolic volume (microM)."""
    c_1: float
    """Ratio between the endoplasmic reticulum (ER) volume and cytosol volume."""

    r_C: float
    r"""Maximum :math:`\text{InsP}_3\text{R}` rate (Hz)."""
    r_L: float
    r"""Maximum :math:`\text{Ca}^{2+}` leak from the ER (Hz)."""

    v_ER: float
    """Maximum rate of SERCA uptake (Hz)."""
    k_ER: float
    r"""SERCA :math:`\text{Ca}^{2+}` affinity (microM)."""

    # [h] equation
    a_2: float
    r""":math:`\text{IP}_3\text{R}` binding rate for :math:`\text{Ca}^{2+}` inhibition.
    (Hz).
    """

    # [InsP3] equation
    v_delta: float
    r"""Maximal rate of :math:`\text{IP}_3` production by PLCδ."""
    k_delta: float
    """Inhibition constant of PLCδ activity."""
    k_PLC: float
    r""":math:`\text{Ca}^{2+}` affinity of PLCδ."""

    v_3K: float
    r"""Maximal rate of degradation by :math:`\text{IP}_3\text{-3K}`."""
    k_D: float
    r""":math:`\text{Ca}^{2+}` affinity of :math:`\text{IP}_3\text{-3K}`."""
    k_3: float
    r""":math:`\text{IP}_3` affinity of :math:`\text{IP}_3\text{-3K}`."""

    r_5P: float
    """Maximal rate of degradation by IP-5P"""


@dataclass(frozen=True)
class DePitta(Model):
    r"""Right-hand side of the model from Equation (14) from [DePitta2009]_.

    This is a 3 equation extensio of :class:`LiRinzel`. The first two equations
    are the same and the third equation describes the evolution of the
    :math:`\text{IP}_3` concentration as

    .. math::

        \frac{\mathrm{d} I}{\mathrm{d} t} =
            \frac{v_\delta}{I + k_\delta} \mathrm{Hill}_2(C, K_{\text{PLC}\delta})
            - v_{3K} \mathrm{Hill}_4(C, K_D) \mathrm{Hill}_1(I, K_3)
            - r_{5P} I.

    .. [DePitta2009] M. De Pittà, M. Goldberg, V. Volman, H. Berry, E. Ben-Jacob,
        *Glutamate Regulation of Calcium and IP3 Oscillating and Pulsating
        Dynamics in Astrocytes*,
        Journal of Biological Physics, Vol. 35, pp. 383--411, 2009,
        `doi:10.1007/s10867-009-9155-y <https://doi.org/10.1007/s10867-009-9155-y>`__.
    """

    param: DePittaParameter
    """Parameters in the De Pittà model."""

    minf: RateFunction
    r"""Steady-state activation function :math:`m_\infty(I)`."""
    ninf: RateFunction
    r"""Steady-state activation function :math:`n_\infty(C)`."""
    Q2: RateFunction
    """Rate function for the inactivation variable :math:`h`."""

    @property
    def variables(self) -> tuple[str, ...]:
        return ("C", "h", "I")

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        C, h, IP3 = args
        param = self.param

        # Equations (1-3) in [DePitta2009].
        C_ER = (param.c_0 - C) / param.c_1
        r_C = param.r_C * param.c_1
        r_L = param.r_L * param.c_1
        v_ER = param.v_ER

        minf = self.minf(IP3)
        ninf = self.ninf(C)

        # Equation (6) in [DePitta2009]
        Q2 = self.Q2(IP3)
        hinf = Q2 / (Q2 + C)
        tau = 1 / (param.a_2 * (Q2 + C))

        # Equation (14) from [DePitta2009]
        k_delta = param.k_delta
        v_delta = param.v_delta * k_delta
        k_PLC = param.k_PLC
        v_3K = param.v_3K
        k_D = param.k_D
        k_3 = param.k_3
        r_5P = param.r_5P

        return (
            r_C * minf**3 * ninf**3 * h**3 * (C_ER - C)
            + r_L * (C_ER - C)
            - v_ER * sym.hill2(C, param.k_ER),
            (hinf - h) / tau,
            v_delta / (IP3 + k_delta) * sym.hill2(C, k_PLC)
            - v_3K * sym.hill4(C, k_D) * sym.hill1(IP3, k_3)
            - r_5P * IP3,
        )


# }}}


# {{{ GlutamateDePitta model


@dataclass(frozen=True)
class GlutamateDePittaParameter(DePittaParameter):
    """Parameters for the glutamate-dependent extension from [DePitta2009]_."""

    beta: float
    """Hill function exponent (should be in :math:`[0.5, 1]`) according to
    Equation (16) in [DePitta2009]_.
    """
    v_beta: float
    """Maximal rate of :math:`\text{IP}_3` production by PLCβ."""
    k_R: float
    """Glutamate affinity of the receptor."""
    k_p: float
    """:math:`\text{Ca}^{2+}`/PKC-dependent inhibition factor."""
    k_pi: float
    """:matr:`\text{Ca}^{2+}` afﬁnity of PKC."""


@dataclass(frozen=True)
class GlutamateDePitta(DePitta):
    r"""An extension of :class:`DePitta` from [DePitta2009]_.

    This model extends :class:`DePitta` by adding another term to the
    :math:`\text{IP}_3` equation. It is given by

    .. math::

        v_\beta \mathrm{Hill}_{\beta}\left(
            \gamma(t),
            K_R + K_p \mathrm{Hill}(C, K_\pi)
        \right).
    """

    param: GlutamateDePittaParameter
    """Parameters in the glutamate-dependent De Pittà model."""

    gamma: Callable[[sym.Expression], sym.Expression]
    """Extracellular glutamate concentration at the astrocytic plasma membrane."""

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        C, h, IP3 = args
        eqs = super().evaluate(t, C, h, IP3)

        # Equation (20) in [DePitta2009]
        param = self.param
        beta = param.beta
        v_beta = param.v_beta
        k_R = param.k_R
        k_p = param.k_p
        k_pi = param.k_pi

        return (
            eqs[0],
            eqs[1],
            v_beta * sym.hill(self.gamma(t), k_R + k_p * sym.hill1(C, k_pi), beta)
            + eqs[2],
        )


# }}}


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
        minf=HillRate(b_1 / a_1, 1),
        ninf=HillRate(b_5 / a_5, 1),
        Q2=LinearRationalRate(b_2 / a_2, b_1 / a_1, b_3 / a_3),
    )


def _make_de_pitta_2009(
    k_ER: float,
    v_delta: float,
    r_5P: float,
    v_beta: float,
    gammas: tuple[float, float],
) -> GlutamateDePitta:
    def gamma(t: sym.Expression) -> sym.Expression:
        from pymbolic.primitives import Comparison, If

        phase = (t % 125.0) / 125.0  # ty: ignore[unsupported-operator]
        return If(Comparison(phase, ">", 0.5), gammas[1], gammas[0])

    param = GlutamateDePittaParameter(
        c_0=2.0,
        c_1=0.185,
        r_C=6.0,
        r_L=0.11,
        v_ER=0.9,
        k_ER=k_ER,
        a_2=0.2,
        v_delta=v_delta,
        k_PLC=0.1,
        k_delta=1.5,
        r_5P=r_5P,
        v_3K=2,
        k_D=0.7,
        k_3=1.0,
        beta=0.7,
        v_beta=v_beta,
        k_R=1.3,
        k_p=10.0,
        k_pi=0.6,
    )

    d1 = 0.13
    d2 = 1.049
    d3 = 0.9434
    d5 = 0.08234

    return GlutamateDePitta(
        param=param,
        minf=HillRate(d1, 1),
        ninf=HillRate(d5, 1),
        Q2=LinearRationalRate(d2, d1, d3),
        gamma=gamma,
    )


ASTROCYTE_MODEL = {
    "LiRinzel1994Figure3": _make_li_rinzel_1994(0.4),
    "DePitta2009Figure12am": _make_de_pitta_2009(0.1, 0.02, 0.04, 0.2, (0.002, 5.0)),
    "DePitta2009Figure12fm": _make_de_pitta_2009(0.05, 0.05, 0.05, 0.5, (0.001, 6.0)),
}


def get_registered_parameters() -> tuple[str, ...]:
    return tuple(ASTROCYTE_MODEL)


def make_model_from_name(name: str) -> Model:
    return ASTROCYTE_MODEL[name]


# }}}
