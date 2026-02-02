# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

from orbitkit.models import Model
from orbitkit.models.rate_functions import RateFunction
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ Astrocyte Calcium Model


@dataclass(frozen=True)
class LiRinzelParameter:
    """Parameters for the astrocyte calcium model from [LiRinzel1994]."""

    C0: float
    """Total free calcium."""
    c1: float
    """Endoplasmic reticulum (ER) to cytosol volume ratio."""
    V_1: float
    V_2: float
    V_3: float
    k3: float

    m: int
    p: int
    q: int
    V_delta: float
    V_ATP: float
    V_5P: float
    V_3K: float
    k_delta: float
    k_ATP: float
    k_5P: float
    k_C: float
    k_3P: float
    D_I: float


@dataclass(frozen=True)
class LiRinzel(Model):
    r"""Right-hand side of an astrocyte calcium model from [LiRinzel1994]_.

    .. math::

        \begin{cases}
        \frac{\mathrm{d} C_i}{\mathrm{d} t} & =
            V_1 m_\infty^3(I_i) n_\infty^3(C_i) h_i^3 (C_{\text{ER}, i} - C_i)
            + V_2 (C_{\text{ER}, i} - C_i)
            - V_3 \frac{C^2_i}{C^2_i + k_3^2}, \\
        \frac{\mathrm{d} I_i}{\mathrm{d} t} & =
            V_\delta \frac{C_i^p}{C_i^p + k_\delta^p}
            + V_{\text{ATP}} \frac{A^m}{A^m + k_{\text{ATP}}^m}
            - V_{5P} \frac{I_i}{I_i + k_{5P}}
            - V_{3K} \frac{C_i^q}{C_i^q + k_{C}^q} \frac{I_i}{I_i + k_{3K}}
            - D_I \sum_{j = 1}^N w_{ij} (I_j - I_i), \\
        \frac{\mathrm{d} h_i}{\mathrm{d} t} & =
            \frac{h_\infty(C_i, I_i) - h_i}{\tau_h(C_i, I_i)},
        \end{cases}

    .. [LiRinzel1994] Y.-X. Li, J. Rinzel,
        *Equations for :math:`\text{InsP}_3` Receptor-Mediated
        :math:`[\text{Ca}^{2+}]_i` Oscillations Derived From a Detailed
        Kinetic Model: A Hodgkin-Huxley Like Formalism*,
        Journal of Theoretical Biology, Vol. 166, pp. 461--473, 1994,
        `doi:10.1006/jtbi.1994.1041 <https://doi.org/10.1006/jtbi.1994.1041>`__.
    """

    param: LiRinzelParameter

    minf: RateFunction
    ninf: RateFunction
    Q2: RateFunction


# }}}
