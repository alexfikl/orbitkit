# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

import orbitkit.symbolic.primitives as sym
from orbitkit.models import Model
from orbitkit.models.rate_functions import RateFunction, SigmoidRate
from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)

# {{{ population


@dataclass(frozen=True)
class WilsonCowanPopulation:
    tau: float
    """Population time scale."""
    r: float
    """Population refractory period."""
    sigmoid: RateFunction
    """Sigmoid activation function."""

    kernels: tuple[sym.DelayKernel, ...]
    r"""Delay kernels :math:`h_{ij}` used in the variables inside the sigmoid."""
    weights: tuple[
        tuple[float | Array | sym.MatrixSymbol, float | Array | sym.MatrixSymbol], ...
    ]
    r"""The weight matrices
    :math:`(\boldsymbol{W}_{*E}^{(k)}, \boldsymbol{W}_{*I}^{(k)}` used in the model.
    """
    forcing: Array | sym.MatrixSymbol
    """Forcing term used in the model."""

    if __debug__:

        def __post_init__(self) -> None:
            if len(self.kernels) != len(self.weights):
                raise ValueError(
                    "kernel and weight counts do not match: "
                    f"{len(self.kernels)} != {len(self.weights)}"
                )

            from orbitkit.symbolic.primitives import MatrixSymbol

            n = self.n
            if isinstance(self.forcing, np.ndarray) and self.forcing.shape != (n,):
                raise ValueError(
                    f"'forcing' has incorrect shape: got {self.forcing.shape} "
                    f"but expected ({n},)"
                )

            for i, (we, wi) in enumerate(self.weights):
                if isinstance(we, (np.ndarray, MatrixSymbol)) and we.shape != (n, n):
                    raise ValueError(
                        f"weight matrix 'W[{i}][0]' has incorrect shape: got "
                        f"{we.shape} but expected ({n}, {n})"
                    )

                if isinstance(wi, (np.ndarray, MatrixSymbol)) and wi.shape != (n, n):
                    raise ValueError(
                        f"weight matrix 'W[{i}][1]' has incorrect shape: got "
                        f"{wi.shape} but expected ({n}, {n})"
                    )

    @property
    def n(self) -> int:
        return self.forcing.shape[0]


# }}}

# {{{ WilsonCowan


@dataclass(frozen=True)
class WilsonCowan(Model):
    r"""Right-hand side of a network Wilson-Cowan model.

    .. math::

        \begin{aligned}
        \tau_E \dot{\boldsymbol{E}} & =
            -\boldsymbol{E} + (1 - r_E \boldsymbol{E}) \boldsymbol{S}_E\left(
                \sum_{k = 1}^K \boldsymbol{W}_{EE}^{(k)} (h^{(k)} \ast \boldsymbol{E})
                - \sum_{k = 1}^K \boldsymbol{W}_{EI}^{(k)} (h^{(k)} \ast \boldsymbol{I})
                + \boldsymbol{P}
            \right), \\
        \tau_I \dot{\boldsymbol{I}} & =
            -\boldsymbol{I} + (1 - r_I \boldsymbol{I}) \boldsymbol{S}_I\left(
                \sum_{k = 1}^K \boldsymbol{W}_{IE}^{(k)} (h^{(k)} \ast \boldsymbol{E})
                - \sum_{k = 1}^K \boldsymbol{W}_{II}^{(k)} (h^{(k)} \ast \boldsymbol{I})
                + \boldsymbol{Q}
            \right),
        \end{aligned}

    where :math:`\boldsymbol{S}_i` are sigmoid activation functions,
    :math:`\boldsymbol{W}^{(k)}_{ij}` are weight matrices with positive entries,
    :math:`(\boldsymbol{P}, \boldsymbol{Q})` are constant forcing terms and
    :math:`h^{(k)}` are distributed delay kernels.
    """

    E: WilsonCowanPopulation
    """Excitatory population parameters."""
    I: WilsonCowanPopulation  # noqa: E741
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

        # compute weighted sums for each term
        terms = []
        for (W_E, W_I), h in zip(self.E.weights, self.E.kernels, strict=True):
            # NOTE: pymbolic should be able to do @ here, but we leave it with
            # a dot product so that it supports scalars
            terms.append(sym.DotProduct(W_E, h(E)) - sym.DotProduct(W_I, h(I)))
        Es = sym.Sum((*terms, self.E.forcing))  # ty: ignore[invalid-argument-type]

        terms = []
        for (W_E, W_I), h in zip(self.I.weights, self.I.kernels, strict=True):
            terms.append(sym.DotProduct(W_E, h(E)) - sym.DotProduct(W_I, h(I)))
        Is = sym.Sum((*terms, self.I.forcing))  # ty: ignore[invalid-argument-type]

        return (
            (-E + (1 - self.E.r * E) * self.E.sigmoid(Es)) / self.E.tau,
            (-I + (1 - self.I.r * I) * self.I.sigmoid(Is)) / self.I.tau,
        )


# }}}


# {{{ parameters


def _make_custom_set1(*, alpha: float = 1) -> WilsonCowan:
    beta = 60
    tau0 = tau1 = 0.5

    Ep = WilsonCowanPopulation(
        tau=1,
        r=0,
        sigmoid=SigmoidRate(1, 0, 1 / beta),
        kernels=(sym.DiracDelayKernel(tau0), sym.DiracDelayKernel(tau1)),
        weights=((-1, 0.4), (np.array([[1.0]]), 0)),
        forcing=np.array([0.65]),
    )
    Ip = WilsonCowanPopulation(
        tau=1 / alpha,
        r=0,
        sigmoid=SigmoidRate(1, 0, 1 / beta),
        kernels=(sym.DiracDelayKernel(tau0),),
        weights=((-1, 0),),
        forcing=np.array([0.5]),
    )

    return WilsonCowan(E=Ep, I=Ip)


def _make_custom_set2(*, alpha: float = 1) -> WilsonCowan:
    # NOTE: this is CoombesLaing2009Figure9 influenced
    beta = 60
    tau0 = tau1 = 0.1

    Ep = WilsonCowanPopulation(
        tau=1,
        r=0,
        sigmoid=SigmoidRate(1, 0, 1 / beta),
        kernels=(sym.DiracDelayKernel(tau0), sym.DiracDelayKernel(tau1)),
        weights=((-6, -2.5), (np.array([[1.0]]), 0)),
        forcing=np.array([0.2]),
    )
    Ip = WilsonCowanPopulation(
        tau=1 / alpha,
        r=0,
        sigmoid=SigmoidRate(1, 0, 1 / beta),
        kernels=(sym.DiracDelayKernel(tau0),),
        weights=((2.5, 6),),
        forcing=np.array([0.2]),
    )

    return WilsonCowan(E=Ep, I=Ip)


def _make_coombes_laing_2009_figure3() -> WilsonCowan:
    # NOTE: should take a small beta to approximate Heaviside function
    beta = 0.01
    tau1 = 1.0
    tau2 = 1.4

    Ep = WilsonCowanPopulation(
        tau=1,
        r=0,
        sigmoid=SigmoidRate(1, 0, beta),
        kernels=(sym.DiracDelayKernel(tau1), sym.DiracDelayKernel(tau2)),
        weights=((-1, 0), (0, 0.4)),
        forcing=np.array([0.7]),
    )
    Ip = WilsonCowanPopulation(
        tau=1,
        r=0,
        sigmoid=SigmoidRate(1, 0, beta),
        kernels=(sym.DiracDelayKernel(tau2), sym.DiracDelayKernel(tau1)),
        weights=((-0.4, 0), (0, 1)),
        forcing=np.array([0.7]),
    )

    return WilsonCowan(E=Ep, I=Ip)


def _make_coombes_laing_2009_figure9(beta: float) -> WilsonCowan:
    tau1 = 0.1
    tau2 = 0.1

    Ep = WilsonCowanPopulation(
        tau=1,
        r=0,
        sigmoid=SigmoidRate(1, 0, 1 / beta),
        kernels=(sym.DiracDelayKernel(tau1), sym.DiracDelayKernel(tau2)),
        weights=((-6, 0), (0, -2.5)),
        forcing=np.array([0.2]),
    )
    Ip = WilsonCowanPopulation(
        tau=1,
        r=0,
        sigmoid=SigmoidRate(1, 0, 1 / beta),
        kernels=(sym.DiracDelayKernel(tau2), sym.DiracDelayKernel(tau1)),
        weights=((2.5, 0), (0, 6)),
        forcing=np.array([0.2]),
    )

    return WilsonCowan(E=Ep, I=Ip)


def _make_muldoon_pasqualetti_2016() -> WilsonCowan:
    se_max = si_max = 1.0

    Ep = WilsonCowanPopulation(
        tau=8,
        r=1 / se_max,
        sigmoid=SigmoidRate(se_max, 4.0, 1 / 1.3),
        kernels=(sym.ZeroDelayKernel(), sym.DiracDelayKernel(10.0)),
        weights=((16.0, 12.0), (np.array([[1.0]]), 0)),
        forcing=np.array([1.25]),
    )
    Ip = WilsonCowanPopulation(
        tau=8,
        r=1 / si_max,
        sigmoid=SigmoidRate(si_max, 3.7, 1 / 2.0),
        kernels=(sym.ZeroDelayKernel(),),
        weights=((15.0, 3.0),),
        forcing=np.array([0.0]),
    )

    return WilsonCowan(E=Ep, I=Ip)


def _make_conti_gorder_2019_figure2ab(tau1: float, tau2: float) -> WilsonCowan:
    alpha = 0.6
    beta = 10.0
    p = q = 0.5
    W = np.array([[0, 1], [1, 0]])

    Ep = WilsonCowanPopulation(
        tau=1,
        r=0,
        sigmoid=SigmoidRate(1, 0, 1 / beta),
        kernels=(
            sym.DiracDelayKernel(tau1),  # E
            sym.DiracDelayKernel(tau2),  # I
            sym.DiracDelayKernel(10.0),  # E
        ),
        weights=(
            (1, 0),  # \tau_{1, n}
            (0, 1),  # \tau_{2, n}
            (W, 0),  # \rho_{nj}
        ),
        forcing=np.array([p, p]),
    )
    Ip = WilsonCowanPopulation(
        tau=1 / alpha,
        r=0,
        sigmoid=SigmoidRate(1, 0, 1 / beta),
        kernels=(
            sym.DiracDelayKernel(tau2),  # E
            sym.DiracDelayKernel(tau1),  # I
        ),
        weights=(
            # NOTE: taking c_ei = 1 does not give the results from Figure 2ab,
            # so after some trial-and-error this seems to work well enough
            (-1, 0),  # \tau_{2, n}
            (0, 1),  # \tau_{1, n}
        ),
        forcing=np.array([q, q]),
    )

    return WilsonCowan(E=Ep, I=Ip)


def _make_conti_gorder_2019_figure2c(tau1: float = 1, tau2: float = 1.4) -> WilsonCowan:
    alpha = 0.6
    beta = 10.0
    p = q = 0.2
    W = np.array([[0, 11], [11, 0]])

    Ep = WilsonCowanPopulation(
        tau=1,
        r=0,
        sigmoid=SigmoidRate(1, 0, 1 / beta),
        kernels=(
            sym.DiracDelayKernel(tau1),
            sym.DiracDelayKernel(tau2),
            sym.DiracDelayKernel(10.0),
        ),
        weights=(
            (-6, 0),  # \tau_{1, n}
            (0, -2.5),  # \tau_{2, n}
            (W, 0),  # \rho_{nj}
        ),
        forcing=np.array([p, p]),
    )
    Ip = WilsonCowanPopulation(
        tau=1 / alpha,
        r=0,
        sigmoid=SigmoidRate(1, 0, 1 / beta),
        kernels=(
            sym.DiracDelayKernel(tau2),  # E
            sym.DiracDelayKernel(tau1),  # I
        ),
        weights=(
            # NOTE: taking c_ei = 2.5 does not give the results from Figure 2ab,
            # so after some trial-and-error this seems to work well enough
            (-2.5, 0),  # \tau_{2, n}
            (0, 6),  # \tau_{1, n}
        ),
        forcing=np.array([q, q]),
    )

    return WilsonCowan(E=Ep, I=Ip)


def _make_conti_gorder_2019_figure345(
    n: int,
    topology: str,
    *,
    k: float,
    rho: float,
    tau1: float = 1,
    tau2: float = 1.4,
) -> WilsonCowan:
    alpha = 0.6
    beta, theta = 10.0, 0.0
    p = 0.5
    q = 0.5

    import orbitkit.adjacency as adj

    dtype = np.float64
    if topology == "bus":
        W = adj.generate_adjacency_bus(n, k=1, dtype=dtype)
    elif topology == "all":
        W = adj.generate_adjacency_all(n, dtype=dtype)
    elif topology == "ring":
        W = adj.generate_adjacency_ring(n, k=1, dtype=dtype)
    elif topology == "lattice":
        W = adj.generate_adjacency_lattice(n, dtype=dtype)
    else:
        raise ValueError(f"unknown adjacency: {topology!r}")

    # NOTE: the paper does not mention normalization, but it's usually done
    W /= np.sum(W, axis=1, keepdims=True)
    assert np.allclose(np.diag(W), 0.0)

    Ep = WilsonCowanPopulation(
        tau=1,
        r=0,
        sigmoid=SigmoidRate(1, theta, 1 / beta),
        kernels=(
            sym.DiracDelayKernel(tau1),
            sym.DiracDelayKernel(tau2),
            sym.DiracDelayKernel(rho),
        ),
        weights=(
            (1, 0),  # \tau_{1, n}
            (0, 1),  # \tau_{2, n}
            (k * W, 0),  # \rho_{nj}
        ),
        forcing=np.full(n, p),
    )
    Ip = WilsonCowanPopulation(
        tau=1 / alpha,
        r=0,
        sigmoid=SigmoidRate(1, theta, 1 / beta),
        kernels=(
            sym.DiracDelayKernel(tau2),  # E
            sym.DiracDelayKernel(tau1),  # I
        ),
        weights=(
            (-1, 0),  # \tau_{2, n}
            (0, 1),  # \tau_{1, n}
        ),
        forcing=np.full(n, q),
    )

    return WilsonCowan(E=Ep, I=Ip)


def _make_conti_gorder_2019_figure3(n: int, topology: str) -> WilsonCowan:
    return _make_conti_gorder_2019_figure345(n, topology, k=1, rho=10)


def _make_conti_gorder_2019_figure4(k: float, topology: str) -> WilsonCowan:
    return _make_conti_gorder_2019_figure345(100, topology, k=k, rho=10)


def _make_conti_gorder_2019_figure5(rho: float, topology: str) -> WilsonCowan:
    return _make_conti_gorder_2019_figure345(100, topology, k=1, rho=rho)


WILSON_COWAN_MODEL = {
    # Others
    "CustomSet1": _make_custom_set1(),
    "CustomSet2": _make_custom_set2(),
    # CoombesLaing2009: https://doi.org/10.1098/rsta.2008.0256
    "CoombesLaing2009Figure3": _make_coombes_laing_2009_figure3(),
    "CoombesLaing2009Figure9a": _make_coombes_laing_2009_figure9(60),
    "CoombesLaing2009Figure9b": _make_coombes_laing_2009_figure9(40),
    # MuldoonPasqualetti2016: https://doi.org/10.1371/journal.pcbi.1005076
    "MuldoonPasqualetti2016": _make_muldoon_pasqualetti_2016(),
    # ContiGorder2019Figure2: https://doi.org/10.1016/j.jtbi.2019.05.010
    "ContiGorder2019Figure2a": _make_conti_gorder_2019_figure2ab(1.0, 1.4),
    "ContiGorder2019Figure2b": _make_conti_gorder_2019_figure2ab(4.0, 40.0),
    "ContiGorder2019Figure2c": _make_conti_gorder_2019_figure2c(),
    # ContiGorder2019Figure3
    "ContiGorder2019Figure3b": _make_conti_gorder_2019_figure3(16, "bus"),
    "ContiGorder2019Figure3c": _make_conti_gorder_2019_figure3(100, "bus"),
    "ContiGorder2019Figure3e": _make_conti_gorder_2019_figure3(16, "all"),
    "ContiGorder2019Figure3f": _make_conti_gorder_2019_figure3(100, "all"),
    "ContiGorder2019Figure3h": _make_conti_gorder_2019_figure3(16, "ring"),
    "ContiGorder2019Figure3i": _make_conti_gorder_2019_figure3(100, "ring"),
    "ContiGorder2019Figure3k": _make_conti_gorder_2019_figure3(16, "lattice"),
    "ContiGorder2019Figure3l": _make_conti_gorder_2019_figure3(100, "lattice"),
    # ContiGorder2019Figure4
    "ContiGorder2019Figure4b": _make_conti_gorder_2019_figure4(1, "bus"),
    "ContiGorder2019Figure4c": _make_conti_gorder_2019_figure4(10, "bus"),
    "ContiGorder2019Figure4e": _make_conti_gorder_2019_figure4(1, "all"),
    "ContiGorder2019Figure4f": _make_conti_gorder_2019_figure4(10, "all"),
    "ContiGorder2019Figure4h": _make_conti_gorder_2019_figure4(1, "ring"),
    "ContiGorder2019Figure4i": _make_conti_gorder_2019_figure4(10, "ring"),
    "ContiGorder2019Figure4k": _make_conti_gorder_2019_figure4(1, "lattice"),
    "ContiGorder2019Figure4l": _make_conti_gorder_2019_figure4(10, "lattice"),
    # ContiGorder2019Figure5
    "ContiGorder2019Figure5b": _make_conti_gorder_2019_figure5(10, "bus"),
    "ContiGorder2019Figure5c": _make_conti_gorder_2019_figure5(25, "bus"),
    "ContiGorder2019Figure5e": _make_conti_gorder_2019_figure5(10, "all"),
    "ContiGorder2019Figure5f": _make_conti_gorder_2019_figure5(25, "all"),
    "ContiGorder2019Figure5h": _make_conti_gorder_2019_figure5(10, "ring"),
    "ContiGorder2019Figure5i": _make_conti_gorder_2019_figure5(25, "ring"),
    "ContiGorder2019Figure5k": _make_conti_gorder_2019_figure5(10, "lattice"),
    "ContiGorder2019Figure5l": _make_conti_gorder_2019_figure5(25, "lattice"),
}


def get_registered_parameters() -> tuple[str, ...]:
    return tuple(WILSON_COWAN_MODEL)


def make_model_from_name(name: str) -> WilsonCowan:
    return WILSON_COWAN_MODEL[name]


# }}}


# {{{ get_wilson_cowan_fixed_points

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


def _get_wilson_cowan_fixed_point(
    sE: RateFunction,
    sI: RateFunction,
    weights: tuple[float, float, float, float],
    forcing: tuple[float, float],
    *,
    bracket: tuple[float, float],
    rtol: float = 1.0e-8,
    method: Methods | None = "brentq",
) -> tuple[float, float] | None:
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

    def root_func_E(E: float, I: float) -> float:  # noqa: E741,N802
        return E - sE_func(a * E - b * I + p)  # ty: ignore[invalid-return-type]

    def root_jac_E(E: float, I: float) -> float:  # noqa: E741,N802
        return 1.0 - a * sE_prime(a * E - b * I + p)

    def solve_for_I(E: float) -> float:  # noqa: N802
        result = so.root_scalar(  # ty: ignore[no-matching-overload]
            lambda x: x - sI_func(c * E - d * x + q),
            fprime=lambda x: 1 + d * sI_prime(c * E - d * x + q),
            method=method,
            bracket=(0, 1),
            x0=0.5,
            x1=0.0,
            rtol=rtol,
        )

        return result.root

    # {{{ root_scalar

    def root_func(E: float) -> float:
        return root_func_E(E, solve_for_I(E))

    def root_jac(E: float) -> float:
        return root_jac_E(E, solve_for_I(E))

    # }}}

    # {{{ minimize_scalar

    def root_func_sqr(E: float) -> float:
        return 0.5 * root_func_E(E, solve_for_I(E)) ** 2

    def root_jac_sqr(E: float) -> float:
        I = solve_for_I(E)  # noqa: E741
        return root_func_E(E, I) * root_jac_E(E, I)

    # }}}

    fa, fb = root_func(bracket[0]), root_func(bracket[1])
    if fa * fb >= 0.0:
        # NOTE: this seems to not converge quite as nicely to the roots, which
        # sometimes leads find more roots that are close but not that close. For
        # safety, we bump the user tolerance a bit and hope for the best.
        rtol *= 1.0e-2

        result = so.minimize_scalar(
            root_func_sqr,
            bracket=bracket,  # ty: ignore[invalid-argument-type]
            tol=rtol,
            options={"xtol": rtol},
            # options={"xatol": 1.0e-4 * rtol},
        )
        if not result.success:
            return None

        # NOTE: we could find a minimum which isn't a root of the equation, in
        # which case we should just call it a failure
        if abs(result.fun) > rtol:
            return None

        E = result.x
    else:
        result = so.root_scalar(  # ty: ignore[no-matching-overload]
            root_func,
            fprime=root_jac,
            method=method,
            bracket=bracket,
            x0=bracket[0],
            x1=bracket[1],
            rtol=rtol,
        )

        if not result.converged:
            return None

        E = result.root

    I = solve_for_I(E)  # noqa: E741
    return E, I


def get_wilson_cowan_fixed_points(
    sE: RateFunction,
    sI: RateFunction,
    weights: tuple[float, float, float, float],
    forcing: tuple[float, float],
    *,
    npoints: int = 32,
    rtol: float = 1.0e-8,
    method: Methods | None = "brentq",
) -> Array:
    r"""Find the synchronized fixed points of the one delay Wilson-Cowan system
    :class:`WilsonCowan`.

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

    This system has between between 1 and 3 real solutions in :math:`(0, 1)`.
    We find these solutions by doing a naive grid search. This method is not
    guaranteed to find all the solutions, but it will always find at least one.
    If more solutions are suspected, choose a finer grid by increasing *npoints*.

    Note that, because the fixed point does not depend on the delay, this
    function can actually compute the fixed points for the whole range of
    models given by the :class:`WilsonCowan` class. In principle, it suffices
    to sum up all the row sums for each variable and put them into :math:`a, b, c`
    or :math:`d`, as appropriate.

    :arg sE: parameters for the sigmoid rate function used in the :math:`E` equation.
    :arg sI: parameters for the sigmoid rate function used in the :math:`I` equation.
    :arg weights: a tuple ``(a, b, c, d)`` for the row sums of all the weight matrices.
    :arg forcing: a tuple ``(p, q)`` for the forcing terms.

    :arg method: one of the methods support by :func:`scipy.optimize.root_scalar`.
    :returns: an array of shape ``(n, 2)`` for each of the fixed points.
    """

    E = np.linspace(0.0, 1.0, npoints)

    fp = []
    for i in range(npoints - 1):
        result = _get_wilson_cowan_fixed_point(
            sE,
            sI,
            weights,
            forcing,
            bracket=(E[i], E[i + 1]),
            rtol=rtol,
            method=method,
        )

        if result is not None:
            fp.append(result)

    result = np.array(fp)

    d = abs(round(np.log10(rtol))) - 2
    _, idx = np.unique(np.round(result, decimals=d), axis=0, return_index=True)

    return result[idx]


# }}}
