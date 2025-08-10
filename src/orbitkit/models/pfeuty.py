# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

import orbitkit.models.symbolic as sym
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
class Pfeuty(sym.Model):
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
        `DOI <https://doi.org/10.3389/neuro.10.008.2007>`__.
    """

    Ainh: Array
    """An adjacency matrix for the inhibitory synaptic current."""
    Agap: Array
    """An adjacency matrix for the electric synaptic current."""
    param: PfeutyParameter
    """Parameters for the Wang-Buzsáki model."""

    alpha_s: sym.RateFunction
    """Normalized concentration of the post-synaptic transmitter-receptor complex."""
    alpha: tuple[sym.RateFunction, sym.RateFunction, sym.RateFunction]
    r"""Rate functions (closed to open) for the Pfeuty model:
    :math:`(\alpha_m, \alpha_h, \alpha_n)`.
    """
    beta: tuple[sym.RateFunction, sym.RateFunction, sym.RateFunction]
    r"""Rate functions (open to closed) for the Pfeuty model:
    :math:(\beta_m, \beta_h, \beta_n).
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

            if isinstance(self.Ainh, np.ndarray) and (
                self.Ainh.ndim != 2 or self.Ainh.shape[0] != self.Ainh.shape[1]
            ):
                raise ValueError(
                    f"adjacency matrix 'Ainh' not square: {self.Ainh.shape}"
                )

            if isinstance(self.Agap, np.ndarray) and (
                self.Agap.ndim != 2 or self.Agap.shape[0] != self.Agap.shape[1]
            ):
                raise ValueError(
                    f"adjacency matrix 'Agap' not square: {self.Agap.shape}"
                )

            if self.Ainh.shape != self.Agap.shape:
                raise ValueError("adjancency matrices have different shapes")

    @property
    def n(self) -> int:
        return 0 if isinstance(self.Ainh, sp.Symbol) else self.Ainh.shape[0]

    @property
    def M_gap(self) -> Array:  # noqa: N802
        """Degree of each of the nodes in the electric synaptic network."""
        return (  # type: ignore[no-any-return]
            sp.Symbol("M_gap")
            if isinstance(self.Agap, sp.Symbol)
            else np.sum(self.Agap, axis=1)
        )

    @property
    def K_inh(self) -> Array:  # noqa: N802
        """Average degree of the nodes in the inhibitory synaptic network."""
        return (  # type: ignore[no-any-return]
            sp.Symbol("K_inh")
            if isinstance(self.Ainh, sp.Symbol)
            else np.mean(np.sum(self.Ainh, axis=1))
        )

    @property
    def K_gap(self) -> Array:  # noqa: N802
        """Average degree of the nodes in the electric synaptic network."""
        return (  # type: ignore[no-any-return]
            sp.Symbol("K_gap")
            if isinstance(self.Agap, sp.Symbol)
            else np.mean(np.sum(self.Agap, axis=1))
        )

    def symbolize(self) -> tuple[Array, tuple[sp.Symbol, ...]]:
        t = sym.var("t")
        Vs = sym.make_sym_vector("Vs", self.n)
        Vd = sym.make_sym_vector("Vd", self.n)
        h = sym.make_sym_vector("h", self.n)
        n = sym.make_sym_vector("n", self.n)
        s = sym.make_sym_vector("s", self.n)

        return self(t, Vs, Vd, h, n, s), (t, Vs, Vd, h, n, s)

    def __call__(
        self, t: float, Vs: Array, Vd: Array, h: Array, n: Array, s: Array
    ) -> Array:
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
        I_inh = self.K_inh * g_inh * (Vd - V_inh) * np.dot(self.Ainh, s)

        # compute the electric synaptic current
        g_gap = param.g_gap
        Is_gap = self.K_gap * g_gap * (self.M_gap * Vs - np.dot(self.Agap, Vs))
        Id_gap = self.K_gap * g_gap * (self.M_gap * Vd - np.dot(self.Agap, Vd))

        # put it all together
        C, tau_inh = param.C, param.tau_inh
        return np.hstack([
            -(Is_L + I_Na + I_K + Is_c + Is_gap) / C,
            -(Id_L + Id_c + I_inh + Id_gap) / C,
            alpha_h * (1 - h) - beta_h * h,
            alpha_n * (1 - n) - beta_n * n,
            alpha_s * (1 - s) - s / tau_inh,
        ])


# }}}


# {{{ parameters from literature

PFEUTY_MODEL = {}


def get_registered_parameters() -> tuple[str, ...]:
    return tuple(PFEUTY_MODEL)


def make_model_from_name(name: str) -> Pfeuty:
    return PFEUTY_MODEL[name]


# }}}
