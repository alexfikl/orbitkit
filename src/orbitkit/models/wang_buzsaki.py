# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from orbitkit.typing import Array
from orbitkit.utils import module_logger

from .common import RateFunction

log = module_logger(__name__)


class WangBuzsakiParameter(NamedTuple):
    C: float
    """Membrane capacitance (micro F/cm^2)."""
    g_Na: float
    r"""Maximum conductance of the :math:`\mathrm{Na}^{+}` current (mS/cm^2)."""
    g_K: float
    r"""Maximum conductance of the :math:`\mathrm{K}^{+}` current (mS/cm^2)."""
    g_L: float
    """Maximum conductance of the leak current (mS/cm^2)."""
    g_syn: float
    """Maximum synaptic conductance (mS/cm^2)."""
    V_Na: float
    r"""Reversal potential for the :math:`\mathrm{N}^{+}` ion channel (mV)."""
    V_K: float
    r"""Reversal potential for the :math:`\mathrm{K}^{+}` ion channel (mV)."""
    V_L: float
    r"""Reversal potential for the leak current (mV)."""
    V_syn: float
    """Reversal potential for the synaptic current (mV)."""
    phi: float
    """Scaling factor for the kinetics of :math:`h` (Hz)."""
    alpha: float
    """Channel opening rate for the synaptic current (Hz)."""
    beta: float
    """Channel closing rate for the synaptic current (Hz)."""


@dataclass(frozen=True)
class WangBuzsaki:
    r"""Right-hand side of the Wang-Buzsáki model from [Wang1996]_.

    .. math::

        \begin{aligned}
        C \frac{\mathrm{d} V}{\mathrm{d} t} & =
            - g_{\text{Na}} m_\infty^2(V) h (V - V_{\text{Na}})
            - g_{\text{K}} n^4 (V - V_{\text{K}})
            - g_{\text{L}} (V - E_{\text{L}})
            - g_{\text{syn}} s (V - V_{\text{syn}}), \\
        \frac{\mathrm{d} h}{\mathrm{d} t} & =
            \phi (\alpha_h(V) (1 - h) - \beta_h(V) h), \\
        \frac{\mathrm{d} n}{\mathrm{d} t} & =
            \phi (\alpha_n(V) (1 - n) - \beta_n(V) (n)), \\
        \frac{\mathrm{d} s}{\mathrm{d} t} & =
            \alpha \alpha(V) (1 - s) - \beta(V) s.
        \end{aligned}
    """

    param: WangBuzsakiParameter
    """Parameters for the Wang-Buzsáki model."""

    minf: RateFunction
    """Steady state for the :math:`m` variable."""
    fpre: RateFunction
    """Normalized concentration of the post-synaptic transmitter-receptor complex."""
    alpha: tuple[RateFunction, RateFunction]
    """Rate functions (closed to open) for the Wang-Buzsáki model."""
    beta: tuple[RateFunction, RateFunction]
    """Rate functions (open to closed) for the Wang-Buzsáki model."""

    def I_app(self, t: float) -> float:  # noqa: N802,PLR6301
        return 0.0

    def __call__(self, t: float, V: Array, h: Array, n: Array, s: Array) -> Array:
        param = self.param

        # compute rate functions
        minf = self.minf(V)
        fpre = self.fpre(V)
        alpha_h, beta_h = self.alpha[0](V), self.beta[0](V)
        alpha_n, beta_n = self.alpha[1](V), self.beta[1](V)

        # compute Na^+ current
        g_Na, V_Na = param.g_Na, param.V_Na
        I_Na = g_Na * minf**2 * h * (V - V_Na)

        # compute K^+ current
        g_K, V_K = param.g_K, param.V_K
        I_K = g_K * n**4 * (V - V_K)

        # compute leak current
        g_L, V_L = param.g_L, param.V_L
        I_L = g_L * (V - V_L)

        # compute synaptic current
        g_syn, V_syn = param.g_syn, param.V_syn
        I_syn = g_syn * (V - V_syn)

        # put it all together
        C, phi, alpha, beta = param.C, param.phi, param.alpha, param.beta
        return np.hstack([
            -(I_Na + I_K + I_L + I_syn + self.I_app(t)) / C,
            phi * (alpha_h * (1 - h) - beta_h * h),
            phi * (alpha_n * (1 - n) - beta_n * n),
            alpha * fpre * (1 - s) - beta * s,
        ])
