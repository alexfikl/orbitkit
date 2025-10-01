# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pymbolic.primitives as prim

import orbitkit.models.symbolic as sym
from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ Wang-Buzsáki Model


@dataclass(frozen=True)
class WangBuzsakiParameter:
    """Parameters for the Wang-Buzsáki model from [Wang1996]_."""

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
    E_Na: float
    r"""Reversal potential for the :math:`\mathrm{N}^{+}` ion channel (mV)."""
    E_K: float
    r"""Reversal potential for the :math:`\mathrm{K}^{+}` ion channel (mV)."""
    E_L: float
    r"""Reversal potential for the leak current (mV)."""
    E_syn: float
    """Reversal potential for the synaptic current (mV)."""
    V_threshold: float
    """Threshold membrane potential (mV)."""
    phi: float
    """Scaling factor for the kinetics of :math:`h` (Hz)."""
    alpha: float
    """Channel opening rate for the synaptic current (Hz)."""
    beta: float
    """Channel closing rate for the synaptic current (Hz)."""


@dataclass(frozen=True)
class WangBuzsaki(sym.Model):
    r"""Right-hand side of the Wang-Buzsáki model from [Wang1996]_.

    .. math::

        \begin{aligned}
        C \frac{\mathrm{d} V_i}{\mathrm{d} t} & =
            - g_{\text{Na}} m_\infty^2(V_i) h_i (V_i - E_{\text{Na}})
            - g_{\text{K}} n_i^4 (V_i - E_{\text{K}})
            - g_{\text{L}} (V_i - E_{\text{L}})
            - g_{\text{syn}} (V_i - E_{\text{syn}}) \sum_{j \ne i} A_{ij} s_j, \\
        \frac{\mathrm{d} h_i}{\mathrm{d} t} & =
            \phi (\alpha_h(V_i) (1 - h_i) - \beta_h(V_i) h_i), \\
        \frac{\mathrm{d} n_i}{\mathrm{d} t} & =
            \phi (\alpha_n(V_i) (1 - n_i) - \beta_n(V_i) n_i), \\
        \frac{\mathrm{d} s_i}{\mathrm{d} t} & =
            \alpha F_{\text{pre}}(V_i) (1 - s_i) - \beta s_i.
        \end{aligned}

    .. [Wang1996] X.-J. Wang, G. Buzsáki,
        *Gamma Oscillation by Synaptic Inhibition in a Hippocampal Interneuronal
        Network Model*,
        The Journal of Neuroscience, Vol. 16, pp. 6402--6413, 1996,
        `doi:10.1523/jneurosci.16-20-06402.1996 <https://doi.org/10.1523/jneurosci.16-20-06402.1996>`__.
    """

    param: WangBuzsakiParameter
    """Parameters for the Wang-Buzsáki model."""
    A: Array | sym.MatrixSymbol
    """An adjacency matrix for the synaptic current."""

    fpre: sym.RateFunction
    """Normalized concentration of the post-synaptic transmitter-receptor complex."""
    alpha: tuple[sym.RateFunction, sym.RateFunction, sym.RateFunction]
    r"""Rate functions (closed to open) for the Wang-Buzsáki model:
    :math:`(\alpha_m, \alpha_h, \alpha_n)`.
    """
    beta: tuple[sym.RateFunction, sym.RateFunction, sym.RateFunction]
    r"""Rate functions (open to closed) for the Wang-Buzsáki model:
    :math:(\beta_m, \beta_h, \beta_n).
    """

    if __debug__:

        def __post_init__(self) -> None:
            assert isinstance(self.param, WangBuzsakiParameter)
            assert isinstance(self.alpha, tuple)
            assert isinstance(self.beta, tuple)

            if len(self.alpha) != 3:
                raise ValueError("must provide exactly 3 'alpha[i]' rate functions")

            if len(self.beta) != 3:
                raise ValueError("must provide exactly 3 'beta[i]' rate functions")

            if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
                raise ValueError(f"adjacency matrix 'A' not square: {self.A.shape}")

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @cached_property
    def M_syn(self) -> Array | sym.MatrixSymbol:  # noqa: N802
        return (
            sym.MatrixSymbol("M_syn", (self.A.shape[0],))
            if isinstance(self.A, sym.MatrixSymbol)
            else np.sum(self.A, axis=1)
        )

    def hinf(self, V: sym.Expression) -> sym.Expression:
        alpha_h, beta_h = self.alpha[1](V), self.beta[1](V)
        return alpha_h / (alpha_h + beta_h)

    def ninf(self, V: sym.Expression) -> sym.Expression:
        alpha_n, beta_n = self.alpha[2](V), self.beta[2](V)
        return alpha_n / (alpha_n + beta_n)

    def sinf(self, V: sym.Expression) -> sym.Expression:
        fpre, alpha, beta = self.fpre(V), self.param.alpha, self.param.beta
        return alpha * fpre / (alpha * fpre + beta)

    @property
    def variables(self) -> tuple[str, ...]:
        return ("V", "h", "n", "s")

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        V, h, n, s = args
        param = self.param

        # compute rate functions
        alpha_m, beta_m = self.alpha[0](V), self.beta[0](V)
        alpha_h, beta_h = self.alpha[1](V), self.beta[1](V)
        alpha_n, beta_n = self.alpha[2](V), self.beta[2](V)
        fpre = self.fpre(V)
        minf = alpha_m / (alpha_m + beta_m)

        # compute Na^+ current
        g_Na, E_Na = param.g_Na, param.E_Na
        I_Na = g_Na * minf**3 * h * (V - E_Na)

        # compute K^+ current
        g_K, E_K = param.g_K, param.E_K
        I_K = g_K * n**4 * (V - E_K)

        # compute leak current
        g_L, E_L = param.g_L, param.E_L
        I_L = g_L * (V - E_L)

        # compute synaptic current
        g_syn, E_syn = param.g_syn, param.E_syn
        I_syn = (
            g_syn * (V - E_syn) * prim.Quotient(sym.DotProduct(self.A, s), self.M_syn)  # type: ignore[arg-type]
        )

        # put it all together
        C, phi, alpha, beta = param.C, param.phi, param.alpha, param.beta
        return (
            -(I_Na + I_K + I_L + I_syn) / C,
            phi * (alpha_h * (1 - h) - beta_h * h),
            phi * (alpha_n * (1 - n) - beta_n * n),
            alpha * fpre * (1 - s) - beta * s,
        )


# }}}

# {{{ Parameters from literature


def _make_wang_buzsaki_1996_model(phi: float = 5.0) -> WangBuzsaki:
    return WangBuzsaki(
        A=np.array([[0, 1], [1, 0]]),
        param=WangBuzsakiParameter(
            C=1.0,
            g_Na=35.0,
            g_K=9.0,
            g_L=0.1,
            g_syn=0.1,
            E_Na=55.0,
            E_K=-90.0,
            E_L=-65.0,
            E_syn=-75.0,
            V_threshold=-52.0,
            phi=phi,
            alpha=12.0,
            beta=0.1,
        ),
        alpha=(
            # alpha_m, alpha_h, alpha_n
            sym.LinearExpm1Rate(0.1, 3.5, -35.0, 10.0),
            sym.ExponentialRate(0.07, -58.0, 20.0),
            sym.LinearExpm1Rate(0.01, 0.34, -34.0, 10.0),
        ),
        beta=(
            # beta_m, beta_h, beta_n
            sym.ExponentialRate(4.0, -60.0, 18.0),
            sym.SigmoidRate(1.0, -28.0, 10.0),
            sym.ExponentialRate(0.125, -44.0, 80.0),
        ),
        fpre=sym.SigmoidRate(1.0, 0.0, 2.0),
    )


WANG_BUZSAKI_MODEL = {
    "WangBuzsaki1996Figure3a": _make_wang_buzsaki_1996_model(5.0),
    "WangBuzsaki1996Figure3b": _make_wang_buzsaki_1996_model(10 / 3),
    "WangBuzsaki1996Figure3c": _make_wang_buzsaki_1996_model(2.0),
}


def get_registered_parameters() -> tuple[str, ...]:
    return tuple(WANG_BUZSAKI_MODEL)


def make_model_from_name(name: str) -> WangBuzsaki:
    return WANG_BUZSAKI_MODEL[name]


# }}}
