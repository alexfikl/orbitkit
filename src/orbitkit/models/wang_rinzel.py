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


# {{{ Wang-Rinzel Model


@dataclass(frozen=True)
class WangRinzelParameter:
    """Parameters for the Wang-Rinzel model from [WangRinzel1992]_."""

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
    """Scaling factor for the kinetics of :math:`h` (Hz)."""


@dataclass(frozen=True)
class WangRinzel(sym.Model):
    r"""Right-hand side of the Wang-Rinzel model from [WangRinzel1992]_.

    .. math::

        \begin{aligned}
        C \frac{\mathrm{d} V_i}{\mathrm{d} t} & =
            - g_{\text{pir}} m^3_\infty(V_i) h_i (V_i - V_{\text{pir}})
            - g_{\text{L}} (V_i - V_{\text{L}})
            - g_{\text{syn}} (V_i - V_{\text{syn}})
                \sum_{j = 0, j \ne i}^N A_{ij} s_\infty(V_j), \\
        \frac{\mathrm{d} h_i}{\mathrm{d} t} & =
            \frac{\phi}{\tau_h(V_i)} (h_\infty(V_i) - h_i).
        \end{aligned}

    where the parameters are described in :class:`WangRinzelParameter`. The
    specific activation functions used in [WangRinzel1992]_ are standard sigmoid
    functions.

    .. [WangRinzel1992] X.-J. Wang, J. Rinzel,
        *Alternating and Synchronous Rhythms in Reciprocally Inhibitory Model Neurons*,
        Neural Computation, Vol. 4, pp. 84--97, 1992,
        `DOI <https://doi.org/10.1162/neco.1992.4.1.84>`__.
    """

    A: Array
    """A connection matrix for the synaptic current."""
    param: WangRinzelParameter
    """Parameters for the Wang-Rinzel model."""

    minf: sym.RateFunction
    r""":math:`m_\infty` activation function used in the membrane potential equation."""
    sinf: sym.RateFunction
    r""":math:`s_\infty` activation function used in the membrane potential equation."""
    hinf: sym.RateFunction
    r""":math:`h_\infty` activation function used in the :math:`h` equation."""
    betah: sym.RateFunction
    r""":math:`\tau_h = h_\infty / \beta_h` activation function used in the
    :math:`h` equation."""

    if __debug__:

        def __post_init__(self) -> None:
            assert isinstance(self.param, WangRinzelParameter)

            if isinstance(self.A, np.ndarray) and (
                self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]
            ):
                raise ValueError(f"adjacency matrix 'A' not square: {self.A.shape}")

    @property
    def n(self) -> int:
        return 0 if isinstance(self.A, sp.Symbol) else self.A.shape[0]

    def symbolize(self) -> tuple[Array, tuple[sp.Symbol, ...]]:
        t = sym.var("t")
        V = sym.make_sym_vector("V", self.n)
        h = sym.make_sym_vector("h", self.n)

        return self(t, V, h), (t, V, h)

    def __call__(self, t: float, V: Array, h: Array) -> Array:
        param = self.param

        # compute activation functions
        minf = self.minf(V)
        sinf = self.sinf(V)
        hinf = self.hinf(V)
        tauh = hinf / self.betah(V)

        # compute PIR current
        g_PIR, V_PIR = param.g_PIR, param.V_PIR
        I_PIR = g_PIR * minf**3 * h * (V - V_PIR)

        # compute leak current
        g_L, V_L = param.g_L, param.V_L
        I_L = g_L * (V - V_L)

        # compute synaptic current
        g_syn, V_syn = param.g_syn, param.V_syn
        I_syn = g_syn * (V - V_syn) * np.dot(self.A, sinf)

        # put it all together and return the right-hand side
        C, phi = param.C, param.phi
        return np.hstack([
            -(I_PIR + I_L + I_syn) / C,
            phi / tauh * (hinf - h),
        ])


# }}}


# {{{ Extended Wang-Rinzel Model


@dataclass(frozen=True)
class WangRinzelExtParameter(WangRinzelParameter):
    """Parameters for the extended Wang-Rinzel model from [WangRinzel1992]_."""

    k_r: float
    """Channel closing rate (Hz)."""


@dataclass(frozen=True)
class WangRinzelExt(WangRinzel):
    r"""Right-hand side of the extended Wang-Rinzel model from [WangRinzel1992]_.

    The extended model has an additional equation for the synaptic variables
    :math:`s`. It is given by:

    .. math::

        \frac{\mathrm{d} s_i}{\mathrm{d} t} = s_\infty(V_i) (1 - s_i) - k_r s_i.
    """

    param: WangRinzelExtParameter
    """Parameters for the extended Wang-Rinzel model."""

    def symbolize(self) -> tuple[Array, tuple[sp.Symbol, ...]]:
        t = sym.var("t")
        V = sym.make_sym_vector("V", self.n)
        h = sym.make_sym_vector("h", self.n)
        s = sym.make_sym_vector("s", self.n)

        return self(t, V, h, s), (t, V, h, s)

    def __call__(self, t: float, V: Array, h: Array, s: Array) -> Array:  # type: ignore[override]
        # FIXME: this is very copy-pasted from the simpler model
        param = self.param

        # compute activation functions
        minf = self.minf(V)
        sinf = self.sinf(V)
        hinf = self.hinf(V)
        tauh = hinf / self.betah(V)

        # compute PIR current
        g_PIR, V_PIR = param.g_PIR, param.V_PIR
        I_PIR = g_PIR * minf**3 * h * (V - V_PIR)

        # compute leak current
        g_L, V_L = param.g_L, param.V_L
        I_L = g_L * (V - V_L)

        # compute synaptic current
        g_syn, V_syn = param.g_syn, param.V_syn
        I_syn = g_syn * (V - V_syn) * np.dot(self.A, s)

        # put it all together and return the right-hand side
        C, phi, k_r = param.C, param.phi, param.k_r
        return np.hstack([
            -(I_PIR + I_L + I_syn) / C,
            phi / tauh * (hinf - h),
            sinf * (1 - s) - k_r * s,
        ])


# }}}


# {{{ Parameters from literature


def _make_wang_rinzel_1992_model(g_PIR: float, theta_syn: float) -> WangRinzel:
    # NOTE: The WangRinzel1992 paper uses `(g_PIR, theta_syn)` as dynamic parameters,
    # where theta_syn is the theta that goes into the `sinf` rate function.

    return WangRinzel(
        # NOTE: WangRinzel1992 uses a simple 2D system of 4 equation
        A=np.array([[0, 1], [1, 0]], dtype=np.int32),
        param=WangRinzelParameter(
            C=1.0,
            g_PIR=g_PIR,
            g_L=0.1,
            g_syn=0.3,
            V_PIR=120.0,
            V_L=-60.0,
            V_syn=-80.0,
            V_threshold=-40.0,
            phi=3.0,
        ),
        minf=sym.SigmoidRate(1.0, -65.0, 7.8),
        sinf=sym.SigmoidRate(1.0, theta_syn, 2.0),
        hinf=sym.SigmoidRate(1.0, -81.0, -11.0),
        betah=sym.ExponentialRate(1.0, -162.3, 17.8),
    )


WANG_RINZEL_MODEL = {
    "Symbolic": WangRinzelExt(
        A=sp.Symbol("A"),
        param=WangRinzelExtParameter(
            C=sym.var("C"),
            g_PIR=sym.var("g_PIR"),
            g_L=sym.var("g_L"),
            g_syn=sym.var("g_syn"),
            V_PIR=sym.var("V_PIR"),
            V_L=sym.var("V_L"),
            V_syn=sym.var("V_syn"),
            V_threshold=-40.0,
            phi=sym.var("phi"),
            k_r=sym.var("k_r"),
        ),
        minf=sp.Function("m_infty"),
        sinf=sp.Function("s_infty"),
        hinf=sp.Function("h_infty"),
        betah=sp.Function("beta_h"),
    ),
    "WangRinzel1992Figure1a": _make_wang_rinzel_1992_model(0.3, -44.0),
    # NOTE: Figure 2 uses g_PIR = 0.3 (like Figure 1) and g_syn = 1.0 (like Figure 3)
    # "WangRinzel1992Figure2a": None,
    # "WangRinzel1992Figure2b": None,
    "WangRinzel1992Figure3a": _make_wang_rinzel_1992_model(1.0, -44.0),
    "WangRinzel1992Figure3c": _make_wang_rinzel_1992_model(1.5, -44.0),
    # NOTE: Figure 4 uses an extended system with an explicit equation for s
    "WangRinzel1992Figure4a": WangRinzelExt(
        A=np.array([[0, 1], [1, 0]], dtype=np.int32),
        param=WangRinzelExtParameter(
            C=1.0,
            g_PIR=0.5,
            g_L=0.05,
            g_syn=0.2,
            V_PIR=120.0,
            V_L=-60.0,
            V_syn=-80.0,
            V_threshold=-40.0,
            phi=2.0,
            k_r=0.005,
        ),
        minf=sym.SigmoidRate(1.0, -65.0, 7.8),
        sinf=sym.SigmoidRate(1.0, -35.0, 2.0),
        hinf=sym.SigmoidRate(1.0, -81.0, -11.0),
        betah=sym.ExponentialRate(1.0, -162.3, 17.8),
    ),
}


def get_registered_parameters() -> tuple[str, ...]:
    return tuple(WANG_RINZEL_MODEL)


def make_model_from_name(name: str) -> WangRinzel:
    return WANG_RINZEL_MODEL[name]


# }}}
