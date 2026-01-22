# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pymbolic.primitives as prim

import orbitkit.symbolic.primitives as sym
from orbitkit.models import Model
from orbitkit.models.rate_functions import (
    ExponentialRate,
    LinearExpm1Rate,
    RateFunction,
    SigmoidRate,
    TanhRate,
)
from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ Pfeuty model


@dataclass(frozen=True)
class PfeutyParameter:
    """Parameters for the Pfeuty model from [Pfeuty2007]_."""

    C: float
    """Membrane capacitance (micro F/cm^2)."""
    g_Na: float
    r"""Maximum conductance of the :math:`\mathrm{Na}^{+}` current (mS/cm^2)."""
    g_K: float
    r"""Maximum conductance of the :math:`\mathrm{K}^{+}` current (mS/cm^2)."""
    g_L: float
    """Maximum conductance of the leak current (mS/cm^2)."""
    g_c: float
    """Maximal axial (coupling) conductance between the soma and dendritic current
    (mS/cm^2)."""
    g_inh: float
    """Maximum inhibitory synaptic conductance (mS/cm^2)."""
    g_gap: float
    """Maximum electric synaptic conductance (mS/cm^2)."""
    V_Na: float
    r"""Reversal potential for the :math:`\mathrm{N}^{+}` ion channel (mV)."""
    V_K: float
    r"""Reversal potential for the :math:`\mathrm{K}^{+}` ion channel (mV)."""
    V_L: float
    r"""Reversal potential for the leak current (mV)."""
    V_inh: float
    """Reversal potential for the inhibitory synaptic current (mV)."""
    V_threshold: float
    """Threshold membrane potential (mV)."""
    tau_inh: float
    """Channel closing time scale for the inhibitory synaptic current (s)."""


@dataclass(frozen=True)
class Pfeuty(Model):
    r"""Right-hand side of the Pfeuty model from [Pfeuty2007]_.

    .. math::

        \begin{aligned}
        C \frac{\mathrm{d} V^\text{s}_i}{\mathrm{d} t} & =
            - g_\text{L} (V^\text{s}_i - V_\text{L})
            - g_{\text{Na}} m^3_\infty h_i (V^\text{s}_i - V_{\text{Na}})
            - g_\text{K} n_i^4 (V^\text{s}_i - V_\text{K})
            - g_\text{c} (V^\text{s}_i - V^\text{d}_i)
            - g_{\text{gap}} \sum_{j \ne i}
                A^{\text{gap}}_{ij} (V^\text{s}_i - V^\text{s}_j), \\\
        C \frac{\mathrm{d} V^\text{d}_i}{\mathrm{d} t} & =
            - g_\text{L} (V^\text{d}_i - V_\text{L})
            - g_\text{c} (V^\text{d}_i - V^\text{s}_i)
            - g_{\text{inh}} (V^\text{d}_i - V_{\text{inh}})
                \sum_{j \ne i} A^{\text{inh}}_{ij} s_j
            - g_{\text{gap}} \sum_{j \ne i}
                A^{\text{gap}}_{ij} (V^\text{d}_i - V^\text{d}_j), \\
        \frac{\mathrm{d} h_i}{\mathrm{d} t} & =
            \alpha_h(V^\text{s}_i) (1 - h_i) - \beta_h(V^\text{s}_i) h_i, \\
        \frac{\mathrm{d} n}{\mathrm{d} t} & =
            \alpha_n(V^\text{s}_i) (1 - n_i) - \beta_n(V^\text{s}_i) n_i, \\
        \frac{\mathrm{d} s}{\mathrm{d} t} & =
            \alpha_s(V^\text{s}_i) (1 - s_i) - s_i / \tau_{\text{inh}}.
        \end{aligned}

    .. [Pfeuty2007] B. Pfeuty,
        *Inhibition Potentiates the Synchronizing Action of Electrical Synapses*,
        Frontiers in Computational Neuroscience, Vol. 1, 2007,
        `doi:10.3389/neuro.10.008.2007 <https://doi.org/10.3389/neuro.10.008.2007>`__.
    """

    param: PfeutyParameter
    """Parameters for the Wang-Buzsáki model."""

    A_inh: Array | sym.MatrixSymbol
    """An adjacency matrix for the inhibitory synaptic current."""
    A_gap: Array | sym.MatrixSymbol
    """An adjacency matrix for the electric synaptic current."""

    alpha_s: RateFunction
    """Normalized concentration of the post-synaptic transmitter-receptor complex."""
    alpha: tuple[RateFunction, RateFunction, RateFunction]
    r"""Rate functions (closed to open) for the Pfeuty model:
    :math:`(\alpha_m, \alpha_h, \alpha_n)`.
    """
    beta: tuple[RateFunction, RateFunction, RateFunction]
    r"""Rate functions (open to closed) for the Pfeuty model:
    :math:`(\beta_m, \beta_h, \beta_n)`.
    """

    if __debug__:

        def __post_init__(self) -> None:
            assert isinstance(self.param, PfeutyParameter)
            assert isinstance(self.alpha, tuple)
            assert isinstance(self.beta, tuple)

            if len(self.alpha) != 3:
                raise ValueError("must provide exactly 3 'alpha[i]' rate functions")

            if len(self.beta) != 3:
                raise ValueError("must provide exactly 3 'beta[i]' rate functions")

            if self.A_inh.ndim != 2 or self.A_inh.shape[0] != self.A_inh.shape[1]:
                raise ValueError(
                    f"adjacency matrix 'A_inh' not square: {self.A_inh.shape}"
                )

            if self.A_gap.ndim != 2 or self.A_gap.shape[0] != self.A_gap.shape[1]:
                raise ValueError(
                    f"adjacency matrix 'A_gap' not square: {self.A_gap.shape}"
                )

            if self.A_inh.shape != self.A_gap.shape:
                raise ValueError("adjacency matrices have different shapes")

    @property
    def n(self) -> int:
        return self.A_inh.shape[0]

    @cached_property
    def M_gap(self) -> Array | sym.MatrixSymbol:  # noqa: N802
        """Degree of each of the nodes in the electric synaptic network."""
        return (
            sym.MatrixSymbol("M_gap", (self.n,))
            if isinstance(self.A_gap, sym.MatrixSymbol)
            else np.sum(self.A_gap, axis=1)
        )

    @property
    def K_inh(self) -> float | sym.Variable:  # noqa: N802
        """Average degree of the nodes in the inhibitory synaptic network."""
        return (
            sym.Variable("K_inh")
            if isinstance(self.A_inh, sym.MatrixSymbol)
            else np.mean(np.sum(self.A_inh, axis=1))
        )

    @property
    def K_gap(self) -> float | sym.Variable:  # noqa: N802
        """Average degree of the nodes in the electric synaptic network."""
        return (
            sym.Variable("K_gap")
            if isinstance(self.A_gap, sym.MatrixSymbol)
            else np.mean(np.sum(self.A_gap, axis=1))
        )

    def hinf(self, V: sym.Expression) -> sym.Expression:
        alpha_h, beta_h = self.alpha[1](V), self.beta[1](V)
        return alpha_h / (alpha_h + beta_h)

    def ninf(self, V: sym.Expression) -> sym.Expression:
        alpha_n, beta_n = self.alpha[2](V), self.beta[2](V)
        return alpha_n / (alpha_n + beta_n)

    def sinf(self, V: sym.Expression) -> sym.Expression:
        alpha, tau_inh = self.alpha_s(V), self.param.tau_inh
        return alpha / (alpha + 1.0 / tau_inh)

    @property
    def variables(self) -> tuple[str, ...]:
        return ("Vs", "Vd", "h", "n", "s")

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        Vs, Vd, h, n, s = args
        param = self.param

        # compute rate functions
        alpha_m, beta_m = self.alpha[0](Vs), self.beta[0](Vs)
        alpha_h, beta_h = self.alpha[1](Vs), self.beta[1](Vs)
        alpha_n, beta_n = self.alpha[2](Vs), self.beta[2](Vs)
        alpha_s = self.alpha_s(Vs)
        minf = alpha_m / (alpha_m + beta_m)

        # compute Na^+ current
        g_Na, V_Na = param.g_Na, param.V_Na
        I_Na = g_Na * minf**3 * h * (Vs - V_Na)

        # compute K^+ current
        g_K, V_K = param.g_K, param.V_K
        I_K = g_K * n**4 * (Vs - V_K)

        # compute leak current
        g_L, V_L = param.g_L, param.V_L
        Is_L = g_L * (Vs - V_L)
        Id_L = g_L * (Vd - V_L)

        # compute coupling current
        g_c = param.g_c
        Is_c = g_c * (Vs - Vd)
        Id_c = -Is_c

        # compute inhibitory synaptic current
        g_inh, V_inh = param.g_inh, param.V_inh
        I_inh = self.K_inh * g_inh * (Vd - V_inh) * sym.DotProduct(self.A_inh, s)

        # compute the electric synaptic current
        # FIXME: if M_gap is an ndarray, its multiplication seems to take precedence
        # and make the whole thing not lazy
        g_gap = param.g_gap
        Is_gap = (
            self.K_gap
            * g_gap
            * (prim.Product((self.M_gap, Vs)) - sym.DotProduct(self.A_gap, Vs))  # ty: ignore[invalid-argument-type]
        )
        Id_gap = (
            self.K_gap
            * g_gap
            * (prim.Product((self.M_gap, Vd)) - sym.DotProduct(self.A_gap, Vd))  # ty: ignore[invalid-argument-type]
        )

        # put it all together
        C, tau_inh = param.C, param.tau_inh
        return (
            -(Is_L + I_Na + I_K + Is_c + Is_gap) / C,
            -(Id_L + Id_c + I_inh + Id_gap) / C,
            alpha_h * (1 - h) - beta_h * h,
            alpha_n * (1 - n) - beta_n * n,
            alpha_s * (1 - s) - s / tau_inh,
        )


# }}}


# {{{ parameters from literature


def _make_pfeuty_2007_model(g_inh: float) -> Pfeuty:
    return Pfeuty(
        A_inh=np.array([[0, 1], [1, 0]]),
        A_gap=np.array([[0, 1], [1, 0]]),
        param=PfeutyParameter(
            C=1.0,
            g_Na=35.0,
            g_K=9.0,
            g_L=0.1,
            g_c=0.3,
            g_inh=g_inh,
            g_gap=0.0,
            V_Na=55.0,
            V_K=-75.0,
            V_L=-65.0,
            V_inh=-75.0,
            # NOTE: this uses the same threshold as Wang-Buzsaki, because Pfeuty
            # does not mention one explicitly and the model is inspired by that
            V_threshold=-52.0,
            tau_inh=3.0,
        ),
        alpha_s=TanhRate(50.0, 0.0, 4.0),
        alpha=(
            # alpha_m, alpha_h, alpha_n
            LinearExpm1Rate(0.1, 3.5, -35.0, 10.0),
            ExponentialRate(0.21, -58.0, 20.0),
            LinearExpm1Rate(0.03, 0.03 * 34.0, -34.0, 10.0),
        ),
        beta=(
            # beta_m, beta_h, beta_n
            ExponentialRate(4.0, -60.0, 18.0),
            SigmoidRate(3.0, -28.0, 10.0),
            ExponentialRate(0.375, -44.0, 80.0),
        ),
    )


PFEUTY_MODEL = {
    "Pfeuty2007Figure2cl": _make_pfeuty_2007_model(0.005),
    "Pfeuty2007Figure2cr": _make_pfeuty_2007_model(0.1),
}


def get_registered_parameters() -> tuple[str, ...]:
    return tuple(PFEUTY_MODEL)


def make_model_from_name(name: str) -> Pfeuty:
    return PFEUTY_MODEL[name]


# }}}
