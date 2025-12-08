# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import orbitkit.symbolic.primitives as sym
from orbitkit.models import Model
from orbitkit.models.rate_functions import RateFunction
from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ DelayedWilsonCowan1


@dataclass(frozen=True)
class WilsonCowanParameter:
    sigmoid: RateFunction
    """Sigmoid activation function."""
    kernels: tuple[sym.DelayKernel, sym.DelayKernel]
    r"""Delay kernels :math:`h_{ij}` used in the variables inside the sigmoid."""
    weights: tuple[Array, Array]
    r"""The weight matrices :math:`\boldsymbol{W}_{ij}` used in the model."""
    forcing: Array
    """Forcing term used in the model."""

    if __debug__:

        def __post_init__(self) -> None:
            assert len(self.kernels) == 2

            if len(self.kernels) != len(self.weights):
                raise ValueError(
                    "kernel and weight counts do not match: "
                    f"{len(self.kernels)} != {len(self.weights)}"
                )

            n = self.n
            if self.forcing.shape != (n,):
                raise ValueError(
                    f"'forcing' has incorrect shape: got {self.forcing.shape} "
                    f"but expected ({n},)"
                )

            for i, w in enumerate(self.weights):
                if w.shape != (n, n):
                    raise ValueError(
                        f"weight matrix '{i}' has incorrect shape: got "
                        f"{w.shape} but expected ({n}, {n})"
                    )

    @property
    def n(self) -> int:
        return self.forcing.shape[0]


@dataclass(frozen=True)
class WilsonCowan1(Model):
    r"""Right-hand side of a network Wilson-Cowan model.

    .. math::

        \begin{aligned}
        \dot{\boldsymbol{E}} & =
            -\boldsymbol{E} + \boldsymbol{S}_E\left(
                \boldsymbol{W}_{00} (h_{00} \ast \boldsymbol{E})
                - \boldsymbol{W}_{01} (h_{01} \ast \boldsymbol{I})
                + \boldsymbol{P}
            \right), \\
        \dot{\boldsymbol{I}} & =
            -\boldsymbol{I} + \boldsymbol{S}_I\left(
                \boldsymbol{W}_{10} (h_{10} \ast \boldsymbol{E})
                - \boldsymbol{W}_{11} (h_{11} \ast \boldsymbol{I})
                + \boldsymbol{Q}
            \right),
        \end{aligned}

    where :math:`\boldsymbol{S}_i` are sigmoid activation functions,
    :math:`\boldsymbol{W}_{ij}` are positive weight matrices, :math:`(\boldsymbol{P},
    \boldsymbol{Q})` are constant forcing terms and :math:`h_{ij}` are distributed
    delay kernels.
    """

    E: WilsonCowanParameter
    """Excitatory population parameters."""
    I: WilsonCowanParameter  # noqa: E741
    """Excitatory population parameters."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.E.n != self.I.n:
                raise ValueError(
                    "'E' and 'I' populations have different sizes: "
                    f"{self.E.n} and {self.I.n}"
                )

    @property
    def n(self) -> int:
        return self.E.n

    @property
    def variables(self) -> tuple[str, ...]:
        return ("E", "I")

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        E, I = args  # noqa: E741

        if E.shape != (self.n,):
            raise ValueError(
                f"'E' shape does not match system: got {E.shape} "
                f"but expected ({self.n},)"
            )

        if I.shape != (self.n,):
            raise ValueError(
                f"'E' shape does not match system: got {E.shape} "
                f"but expected ({self.n},)"
            )

        # unpack variables
        W_EE, W_EI = self.E.weights
        W_IE, W_II = self.I.weights
        P, Q = self.E.forcing, self.I.forcing

        h_EE, h_EI = self.E.kernels
        h_IE, h_II = self.I.kernels

        return (
            -E + self.E.sigmoid(W_EE * h_EE(E) - W_EI * h_EI(I) + P),
            -I + self.I.sigmoid(W_IE * h_IE(E) - W_II * h_II(I) + Q),
        )


# }}}
