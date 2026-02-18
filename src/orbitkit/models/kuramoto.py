# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import orbitkit.symbolic.primitives as sym
from orbitkit.models import Model
from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


def shift_kuramoto_angle(theta: Array) -> Array:
    # FIXME: there's probably faster way to do this, but this is nice and clear
    return np.angle(np.exp(1j * theta))


# {{{ Kuramoto


@dataclass(frozen=True)
class Kuramoto(Model):
    r"""Right-hand side of the classic Kuramoto model from [Kuramoto1984]_.

    .. math::

        \frac{\mathrm{d} \theta_i}{\mathrm{d} t} =
            \omega_i + \sum_{j = 0}^{N - 1}
                K_{ij} \sin (\theta_j - \theta_i - \alpha),

    where we note that the sum is not averaged by the number of nodes. If such a
    weighting is desired, it should be added into the weight matrix :math:`K`.

    .. [Kuramoto1984] Y. Kuramoto,
        *Chemical Oscillations, Waves, and Turbulence*,
        Springer Berlin Heidelberg, 1984.
        `doi:10.1007/978-3-642-69689-3 <https://doi.org/10.1007/978-3-642-69689-3>`__.
    """

    omega: Array | sym.Expression
    """Frequency of ech oscillator in the Kuramoto model."""
    alpha: sym.Expression
    """Phase lag for each oscillator in the Kuramoto model."""
    K: Array | sym.MatrixSymbol
    """Coupling matrix between the different populations."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.K.ndim != 2 or self.K.shape[0] != self.K.shape[1]:
                raise ValueError(f"coupling matrix 'K' must be square: {self.K.shape}")

            from numbers import Real

            pi2 = np.pi / 2.0
            if isinstance(self.alpha, Real) and not -pi2 <= float(self.alpha) <= pi2:
                raise ValueError(
                    f"phase lag 'alpha' must be in [-pi/2, pi/2]: {self.alpha}"
                )

    @property
    def n(self) -> int:
        return self.K.shape[0]

    @property
    def variables(self) -> tuple[str, ...]:
        return ("theta",)

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        (theta,) = args
        K = self.K

        if theta.shape != (K.shape[0],):
            raise ValueError(
                "'theta' shape does not match coupling 'K': "
                f"got {theta.shape} but expected ({K.shape[0]},)"
            )

        result = sym.Sum((  # ty: ignore[invalid-argument-type]
            self.omega,
            sym.Contract(
                sym.Product((  # ty: ignore[invalid-argument-type]
                    self.K,
                    sym.sin(theta.reshape(1, -1) - theta.reshape(-1, 1) - self.alpha),
                )),
                axes=(1,),
            ),
        ))

        return (result,)


# }}}

# {{{ KuramotoAbrams


@dataclass(frozen=True)
class KuramotoAbrams(Model):
    r"""Right-hand side of the Kuramoto-like Abrams model from [Abrams2008]_.

    .. math::

        \frac{\mathrm{d} \theta^\sigma_i}{\mathrm{d} t} =
            \omega
            + \sum_{\sigma' = 0}^{m - 1} \frac{K_{\sigma \sigma'}}{n_{\sigma'}}
                \sum_{j = 0}^{n_{\sigma'} - 1}
                \sin (\theta^{\sigma'}_j - \theta^\sigma_i - \alpha)

    The model has :math:`m` identical populations, each with :math:`n_\sigma`
    nodes. The coupling between the populations is given by the :math:`K` matrix.

    .. [Abrams2008] D. M. Abrams, R. Mirollo, S. H. Strogatz, D. A. Wiley,
        *Solvable Model for Chimera States of Coupled Oscillators*,
        Physical Review Letters, Vol. 101, pp. 84103--84103, 2008,
        `doi:10.1103/physrevlett.101.084103 <https://doi.org/10.1103/physrevlett.101.084103>`__.
    """

    omega: sym.Expression
    """Frequency of each oscillator in the Kuramoto model."""
    alpha: sym.Expression
    """Phase lag for each oscillator in the Kuramoto model."""
    K: Array | sym.MatrixSymbol
    """Coupling matrix between the different populations."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.K.ndim != 2 or self.K.shape[0] != self.K.shape[1]:
                raise ValueError(f"coupling matrix 'K' must be square: {self.K.shape}")

            from numbers import Real

            if isinstance(self.omega, Real) and self.omega < 0:
                raise ValueError(
                    f"frequency 'omega' must be non-negative: {self.omega}"
                )

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(f"theta{i}" for i in range(self.K.shape[0]))

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        thetas = args
        K = self.K

        if K.shape[0] != len(thetas):
            raise ValueError(
                "coupling matrix size does not match number of populations: "
                f"matrix has shape {K.shape} for {len(thetas)} populations"
            )

        from pymbolic.primitives import flattened_sum

        return tuple(
            # FIXME: we already flatten in the codegen, so this is probably not needed
            flattened_sum([
                self.omega
                + sum(
                    K[a, b]
                    / theta_b.shape[0]
                    * sym.Contract(
                        sym.sin(
                            theta_b.reshape(-1, 1) - theta_a.reshape(1, -1) - self.alpha
                        ),
                        axes=(0,),
                    )
                    for b, theta_b in enumerate(thetas)
                )
            ])
            for a, theta_a in enumerate(thetas)
        )


# }}}


# {{{ Parameters from literature


def _make_kuramoto_schroder_2017_model(K: float) -> Kuramoto:
    #                 0     1     2     3    4    5    6     7     8    9
    omega = np.array([0.5, -1.0, -0.2, -0.5, 0.3, 0.1, 0.9, -0.8, -0.3, 1.0])
    A = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    return Kuramoto(omega=omega, alpha=0.0, K=K * (A | A.T))


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
    "Schroder2017Figure1b": _make_kuramoto_schroder_2017_model(0.05),
    "Schroder2017Figure1c": _make_kuramoto_schroder_2017_model(0.25),
    "Schroder2017Figure1d": _make_kuramoto_schroder_2017_model(0.50),
    "Schroder2017Figure1e": _make_kuramoto_schroder_2017_model(1.50),
    "Schroder2017Figure1f": _make_kuramoto_schroder_2017_model(2.00),
    "Schroder2017Figure1g": _make_kuramoto_schroder_2017_model(10.0),
    "Abrams2008Figure1": _make_kuramoto_abrams_2008_model(0.1, 0.2),
    "Abrams2008Figure2a": _make_kuramoto_abrams_2008_model(0.1, 0.2),
    "Abrams2008Figure2b": _make_kuramoto_abrams_2008_model(0.1, 0.28),
    "Abrams2008Figure2c": _make_kuramoto_abrams_2008_model(0.1, 0.35),
}


def get_registered_parameters() -> tuple[str, ...]:
    return tuple(KURAMOTO_MODEL)


def make_model_from_name(name: str) -> Model:
    return KURAMOTO_MODEL[name]


# }}}
