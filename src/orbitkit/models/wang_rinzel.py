# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, replace

from orbitkit.typing import Array
from orbitkit.utils import module_logger

from .common import ExponentialRate, RateFunction, SigmoidRate

log = module_logger(__name__)


@dataclass(frozen=True)
class WangRinzelParameter:
    C: float
    """Membrane capacitance (micro F/cm^2)."""
    g_PIR: float
    r"""Maximum conductance of the PIR current (mS/cm^2)."""
    g_L: float
    """Maximum conductance of the leak current (mS/cm^2)."""
    g_syn: float
    """Maximum synaptic conductance (mS/cm^2)."""
    V_PIR: float
    r"""Reversal potential for the PIR current (mV)."""
    V_L: float
    r"""Reversal potential for the leak current (mV)."""
    V_syn: float
    """Reversal potential for the synaptic current (mV)."""
    V_threshold: float
    """Threshold membrane potential (mV)."""
    phi: float
    """Scaling factor for the kinetics of :math:`h` (1/s)."""


@dataclass(frozen=True)
class WangRinzel:
    r"""Right-hand side of the Wang-Rinzel model from [Wang1992]_.

    .. math::

        \begin{aligned}
        C \frac{\mathrm{d} V_i}{\mathrm{d} t} & =
            - g_{\text{pir}} m^3_\infty(V_i) h_i (V_i - V_{\text{pir}})
            - g_{\text{L}} (V_i - V_{\text{L}})
            - g_{\text{syn}} s_\infty(V_i) (V_i - V_{\text{syn}}), \\
        \frac{\mathrm{d} h_i}{\mathrm{d} t} & =
            \frac{\phi}{\tau_h(V_i)} (h_\infty(V_i) - h_i).
        \end{aligned}

    where the parameters are described in :class:`WangRinzelParameter`. The
    specific activation functions used in [Wang1992]_ are standard sigmoid
    functions.

    .. [Wang1992] X.-J. Wang, J. Rinzel,
        *Alternating and Synchronous Rhythms in Reciprocally Inhibitory Model Neurons*,
        Neural Computation, Vol. 4, pp. 84--97, 1992,
        `DOI <https://doi.org/10.1162/neco.1992.4.1.84>`__.
    """

    param: WangRinzelParameter
    """Parameers for the Wang-Rinzel model."""
    minf: RateFunction
    r""":math:`m_\infty` activation function used in the membrane potential equation."""
    sinf: RateFunction
    r""":math:`s_\infty` activation function used in the membrane potential equation."""
    hinf: RateFunction
    r""":math:`h_\infty` activation function used in the :math:`h` equation."""
    betah: RateFunction
    r""":math:`\tau_h = h_\infty / \beta_h` activation function used in the
    :math:`h` equation."""

    def __call__(self, t: float, V: Array, h: Array) -> tuple[Array, Array]:
        param = self.param

        # compute activation functions
        minf = self.minf(V)
        sinf = self.sinf(V)
        hinf = self.hinf(V)
        tauh = hinf / self.betah(V)

        # compute PIR current
        g_PIR, V_PIR = param.g_PIR, param.V_PIR
        I_PIR = -g_PIR * minf**3 * h * (V - V_PIR)

        # compute leak current
        g_L, V_L = param.g_L, param.V_L
        I_L = -g_L * (V - V_L)

        # compute synaptic current
        g_syn, V_syn = param.g_syn, param.V_syn
        I_syn = -g_syn * sinf * (V - V_syn)

        # put it all together and return the right-hand side
        C, phi = param.C, param.phi
        return (
            (I_PIR + I_L + I_syn) / C,
            phi / tauh * (hinf - h),
        )


# {{{ parameters from literature

# NOTE: The WangRinzel1992 paper uses `(g_syn, theta_syn)` as dynamic parameters,
# where theta_syn is the theta that goes into the `sinf` rate function. The
# examples below correspond to some of the figures in the papers.
_WANG_RINZEL_1992_PARAMETERS = WangRinzelParameter(
    C=1.0,
    g_PIR=0.3,
    g_L=0.1,
    g_syn=0.3,
    V_PIR=120.0,
    V_L=-60.0,
    V_syn=-80.0,
    V_threshold=-40.0,
    phi=3.0,
)

_WANG_RINZEL_1992_MODEL = WangRinzel(
    param=_WANG_RINZEL_1992_PARAMETERS,
    minf=SigmoidRate(1.0, -65.0, 7.8),
    sinf=SigmoidRate(1.0, -44.0, 2.0),
    hinf=SigmoidRate(1.0, -81.0, -11.0),
    betah=ExponentialRate(1.0, -162.3, 17.8),
)

WANG_RINZEL_PARAMETERS = {
    "WangRinzel1992Figure1a": _WANG_RINZEL_1992_PARAMETERS,
    "WangRinzel1992Figure3a": replace(_WANG_RINZEL_1992_PARAMETERS, g_syn=1.0),
    "WangRinzel1992Figure3c": replace(_WANG_RINZEL_1992_PARAMETERS, g_syn=1.5),
    "WangRinzel1992Figure4a": replace(
        _WANG_RINZEL_1992_PARAMETERS, g_PIR=0.5, g_syn=0.2, g_L=0.05, phi=2.0
    ),
}


WANG_RINZEL_MODEL = {
    "WangRinzel1992Figure1a": _WANG_RINZEL_1992_MODEL,
    "WangRinzel1992Figure3a": replace(
        _WANG_RINZEL_1992_MODEL,
        param=WANG_RINZEL_PARAMETERS["WangRinzel1992Figure3a"],
    ),
    "WangRinzel1992Figure3c": replace(
        _WANG_RINZEL_1992_MODEL,
        param=WANG_RINZEL_PARAMETERS["WangRinzel1992Figure3c"],
    ),
    "WangRinzel1992Figure4a": replace(
        _WANG_RINZEL_1992_MODEL,
        param=WANG_RINZEL_PARAMETERS["WangRinzel1992Figure4a"],
        sinf=SigmoidRate(1.0, -35.0, 0.005),
    ),
}


# }}}
