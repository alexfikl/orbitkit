# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import orbitkit.symbolic.primitives as sym
from orbitkit.models import Model
from orbitkit.models.rate_functions import RateFunction
from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ WilsonCowan1


@dataclass(frozen=True)
class WilsonCowanPopulation:
    sigmoid: RateFunction
    """Sigmoid activation function."""
    kernels: tuple[sym.DelayKernel, ...]
    r"""Delay kernels :math:`h_{ij}` used in the variables inside the sigmoid."""
    weights: tuple[Array, ...]
    r"""The weight matrices :math:`\boldsymbol{W}_{ij}` used in the model."""
    forcing: Array
    """Forcing term used in the model."""

    if __debug__:

        def __post_init__(self) -> None:
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

    E: WilsonCowanPopulation
    """Excitatory population parameters."""
    I: WilsonCowanPopulation  # noqa: E741
    """Excitatory population parameters."""

    if __debug__:

        def __post_init__(self) -> None:
            if len(self.E.kernels) != 2 or len(self.I.kernels) != 2:
                raise ValueError(
                    "Expected only two kernels for this model: "
                    f"got {len(self.E.kernels)} excitatory and "
                    f"{len(self.I.kernels)} inhibitory kernels"
                )

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

    @property
    def rattrs(self) -> set[str]:
        return {"E", "I", "kernels", "weights"}

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


# {{{

# FIXME: would be nice if scipy offered these
Methods: TypeAlias = Literal[
    "bisect",
    "brentq",
    "brenth",
    "ridder",
    "toms748",
    "newton",
    "secant",
    "halley",
]


def get_wilson_cowan_fixed_point(
    sE: RateFunction,
    sI: RateFunction,
    weights: tuple[float, float, float, float],
    forcing: tuple[float, float],
    *,
    rtol: float = 1.0e-8,
    method: Methods | None = None,
) -> tuple[float, float]:
    r"""Find a synchronized fixed point of the one delay Wilson-Cowan system
    :class:`WilsonCowan1`.

    To find a synchronized fixed point, we assume that all the weight matrices
    have equal row sums, given by the *weights* tuple. We also assume that the
    forcing term is uniform and is given by the *forcing* tuple. Under these
    assumptions and regardless of the delay, we have that a fixed point of the
    system must satisfy:

    .. math::

        \begin{aligned}
            E^\star & = S_E(a E^\star - b I^\star + p), \\
            I^\star & = S_I(c I^\star - d I^\star + q).
        \end{aligned}

    This system has at least one real solution in :math:`(0, 1)` (and at most 3
    such solutions).

    :arg sE: parameters for the sigmoid rate function used in the :math:`E` equation.
    :arg sI: parameters for the sigmoid rate function used in the :math:`I` equation.
    :arg weights: a tuple ``(a, b, c, d)`` for the row sums of all the weight matrices.
    :arg forcing: a tuple ``(p, q)`` for the forcing terms.

    :arg method: one of the methods support by :func:`scipy.optimize.root_scalar`.
    :returns: a fixed point of the system as a tuple ``(Estar, Istar)``.
    """
    from orbitkit.codegen import lambdify

    x = sym.Variable("x")
    sE_func, sE_prime = lambdify(x, sE(x)), lambdify(x, sE.diff(x))
    sI_func, sI_prime = lambdify(x, sI(x)), lambdify(x, sI.diff(x))

    a, b, c, d = weights
    p, q = forcing

    # NOTE: We essentially have two equations here
    #
    #   E = S_E(a E - b I + p)
    #   I = S_I(c E - d I + q)
    #
    # which we solve by nested 1d root finding. This should be pretty robust and
    # lets us take advantage of two properties of our problem:
    #
    #   1. We know that the solutions are in (0, 1)
    #   2. We know that the sigmoids are nice and increasing.
    #
    # FIXME: This problem can have 1 or 3 solutions, depending on how the lines
    # intersect the sigmoid. This function only finds one of them, which is not
    # great. To find more, we could
    #   * do a bit of analysis to see when this is the case.
    #   * better bracket the solutions?

    import scipy.optimize as so

    def solve_for_i(E: float) -> float:
        result = so.root_scalar(
            lambda x: x - sI_func(c * E - d * x + q),
            method=method,  # ty: ignore[invalid-argument-type]
            fprime=lambda x: 1 + sI_prime(c * E - d * x + q) * d,  # ty: ignore[invalid-argument-type]
            bracket=(0, 1),
            rtol=rtol,
        )

        return result.root

    def root_func(E: float) -> float:
        I = solve_for_i(E)  # noqa: E741
        return E - sE_func(a * E - b * I + p)  # ty: ignore[invalid-return-type]

    def root_jac(E: float) -> float:
        I = solve_for_i(E)  # noqa: E741
        return 1.0 + sE_prime(a * E - b * I + p) * a

    result = so.root_scalar(
        root_func,
        method=method,  # ty: ignore[invalid-argument-type]
        fprime=root_jac,  # ty: ignore[invalid-argument-type]
        bracket=(0, 1),
        rtol=rtol,
    )
    E = result.root
    I = solve_for_i(E)  # noqa: E741

    return E, I


# }}}
