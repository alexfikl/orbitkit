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


@dataclass(frozen=True)
class KuramotoAbrams(sym.Model):
    r"""Right-hand side of the Adams model from [Abrams2008]_.

    .. math::

        \frac{\mathrm{d} \theta^\sigma_i}{\mathrm{d} t} =
            \omega
            + \sum_{\sigma' = 0}^{m - 1} \frac{K_{\sigma \sigma'}}{n_\sigma'}
                \sum_{j = 0}^{n_\sigma' - 1}
                \sin (\theta^{\sigma'}_j - \theta^\sigma_i - \alpha)

    The model has :math:`m` identical populations, each with :math:`n_\sigma`
    nodes. The coupling between the populations is given by the :math:`K` matrix.

    .. [Abrams2008] D. M. Abrams, R. Mirollo, S. H. Strogatz, D. A. Wiley,
        *Solvable Model for Chimera States of Coupled Oscillators*,
        Physical Review Letters, Vol. 101, pp. 84103--84103, 2008,
        `doi:10.1103/physrevlett.101.084103 <https://doi.org/10.1103/physrevlett.101.084103>`__.
    """

    omega: float
    """Frequency of each oscillator in the Kuramoto model."""
    alpha: float
    """Phase lag for each oscillator in the Kuramoto model."""
    K: Array
    """Coupling matrix between the different populations."""

    if __debug__:

        def __post_init__(self) -> None:
            if isinstance(self.K, np.ndarray) and (
                self.K.ndim != 2 or self.K.shape[0] != self.K.shape[1]
            ):
                raise ValueError(f"coupling matrix 'K' must be square: {self.K.shape}")

            from numbers import Number

            if isinstance(self.omega, Number) and self.omega < 0:
                raise ValueError(
                    f"frequency 'omega' must be non-negative: {self.omega}"
                )

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(f"theta{i}" for i in range(self.K.shape[0]))

    def evaluate(self, t: float, *args: Array) -> Array:
        thetas = args
        K = self.K

        if K.shape[0] != len(thetas):
            raise ValueError(
                "coupling matrix size does not match number of populations: "
                f"matrix has shape {K.shape} for {len(thetas)} populations"
            )

        omega, alpha = self.omega, self.alpha
        if isinstance(thetas[0], np.ndarray):
            return np.hstack([
                omega
                + sum(
                    K[a, b]
                    / theta_b.size
                    * sum(
                        sym.vectorize(sp.sin, theta_j - theta_a - alpha)
                        for theta_j in theta_b
                    )
                    for b, theta_b in enumerate(thetas)
                )
                for a, theta_a in enumerate(thetas)
            ])
        else:
            # FIXME: this is mostly for pretty printing the symbolic expressions.
            # It would be nice to have a more general expression that just works
            # in both cases like for the other models..
            j = sp.Idx("j")
            n = [sp.Symbol(f"n{i}") for i in range(len(thetas))]

            # FIXME: sympy does not print the loop index if each theta is a
            # separate IndexedBase. Not clear why that would be..
            theta = sp.IndexedBase("theta")

            return np.hstack([
                omega
                + sum(
                    K[a, b]
                    / n[b]
                    * sp.Sum(
                        sp.sin(theta[b, j] - theta[a] - alpha),
                        (j, 0, n[b] - 1),
                    )
                    for b, theta_b in enumerate(thetas)
                )
                for a, theta_a in enumerate(thetas)
            ])


def _make_kuramoto_abrams_2008_model(beta: float, A: float) -> KuramotoAbrams:
    alpha = np.pi / 2.0 - beta
    mu = (1 + A) / 2.0
    nu = (1 - A) / 2.0

    return KuramotoAbrams(
        # NOTE: omega is not mentioned explicitly in the paper, but it does not
        # seem like it matters for the results that are presented (i.e. it's
        # just an additive constant that will get averaged out anyway)
        omega=0.0,
        alpha=alpha,
        K=np.array([[mu, nu], [nu, mu]]),
    )


KURAMOTO_MODEL = {
    "Abrams2008Figure1": _make_kuramoto_abrams_2008_model(0.1, 0.2),
    "Abrams2008Figure2a": _make_kuramoto_abrams_2008_model(0.1, 0.2),
    "Abrams2008Figure2b": _make_kuramoto_abrams_2008_model(0.1, 0.28),
    "Abrams2008Figure2c": _make_kuramoto_abrams_2008_model(0.1, 0.35),
}


def make_model_from_name(name: str) -> KuramotoAbrams:
    return KURAMOTO_MODEL[name]
