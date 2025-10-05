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


# {{{ FitzHughNagumoOmelchenko


@dataclass(frozen=True)
class FitzHughNagumoOmelchenko(sym.Model):
    r"""Right-hand side of a network FitzHugh-Nagumo model from [Omelchenko2013]_.

    .. math::

        \begin{aligned}
        \epsilon \frac{\mathrm{d} u_i}{\mathrm{d} t} & =
            u_i - \frac{u_i^3}{3} - v_i
            + \frac{\sigma}{g_i} \sum_{j = 0}^n G_{ij} H_{uu} (u_j - u_i)
            + \frac{\sigma}{g_i} \sum_{j = 0}^n G_{ij} H_{uv} (v_j - v_i), \\
        \frac{\mathrm{d} v_i}{\mathrm{d} t} & =
            u + a
            + \frac{\sigma}{g_i} \sum_{j = 0}^n G_{ij} H_{vu} (u_j - u_i)
            + \frac{\sigma}{g_i} \sum_{j = 0}^n G_{ij} H_{vv} (v_j - v_i),
        \end{aligned}

    where :math:`\epsilon` is a time scale, :math:`\sigma` is a coupling strength,
    and :math:`g_i` is the degree of node :math:`i` based on the adjacency matrix
    :math:`G_{ij}`.

    .. [Omelchenko2013] I. Omelchenko, O. E. Omel'chenko, P. Hövel, E. Schöll,
        *When Nonlocal Coupling Between Oscillators Becomes Stronger:
        Patched Synchrony or Multichimera States*,
        Physical Review Letters, Vol. 110, pp. 224101--224101, 2013,
        `doi:10.1103/physrevlett.110.224101 <https://doi.org/10.1103/physrevlett.110.224101>`__.
    """

    a: sym.Expression
    """Parameter in the FitzHugh-Nagumo model."""
    epsilon: sym.Expression
    """Parameter that characterizes the time scale separation between the components."""
    sigma: sym.Expression
    """Coupling strength between the nodes."""

    H: Array | sym.MatrixSymbol
    """Matrix controlling the coupling between the components."""
    G: Array | sym.MatrixSymbol
    """Adjacency matrix that describes the inter-node interactions in each component."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.H.shape != (2, 2):
                raise ValueError(f"coupling matrix 'H' must be (2, 2): {self.H.shape}")

            if self.G.ndim != 2 or self.G.shape[0] != self.G.shape[1]:
                raise ValueError(f"adjacency matrix 'G' must be square: {self.G.shape}")

    @property
    def n(self) -> int:
        return self.G.shape[0]

    @property
    def variables(self) -> tuple[str, ...]:
        return ("u", "v")

    @cached_property
    def g(self) -> Array | sym.MatrixSymbol:
        """Degree of each of the nodes in the electric synaptic network."""
        return (
            sym.MatrixSymbol("g", (self.n,))
            if isinstance(self.G, sym.MatrixSymbol)
            else np.sum(self.G, axis=1)
        )

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        u, v = args

        if u.shape != (self.n,):
            raise ValueError(
                "'u' shape does not match adjacency matrix: "
                f"got {u.shape} but expected ({self.n},)"
            )

        if v.shape != (self.n,):
            raise ValueError(
                "'v' shape does not match adjacency matrix: "
                f"got {v.shape} but expected ({self.n},)"
            )

        def sum_g(a: sym.Expression, b: sym.Expression) -> sym.Expression:
            return sym.Contract(
                prim.Product((  # type: ignore[arg-type]
                    self.G,
                    a * (u.reshape(1, -1) - u.reshape(-1, 1))
                    + b * (v.reshape(1, -1) - v.reshape(-1, 1)),
                )),
                axes=(1,),
            )

        H, epsilon, a = self.H, self.epsilon, self.a
        sigma_g = prim.Quotient(self.sigma, self.g)  # type: ignore[arg-type]
        return (
            (u - u**3 / 3 - v + sigma_g * sum_g(H[0, 0], H[0, 1])) / epsilon,
            u + a + sigma_g * sum_g(H[1, 0], H[1, 1]),
        )


# }}}


# {{{


def _make_fhn_omelchenko_2019(sigma: float) -> FitzHughNagumoOmelchenko:
    from orbitkit.adjacency import generate_adjacency_fractal

    G = generate_adjacency_fractal("110011", nlevels=4)

    phi = np.pi - 0.1

    return FitzHughNagumoOmelchenko(
        a=0.5,
        epsilon=0.05,
        sigma=sigma,
        H=np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]]),
        G=G,
    )


FITZHUGH_NAGUMO_MODEL = {
    "Omelchenko2019Figure1": _make_fhn_omelchenko_2019(0.05),
    "Omelchenko2019Figure4a": _make_fhn_omelchenko_2019(0.1),
    "Omelchenko2019Figure4b": _make_fhn_omelchenko_2019(0.15),
}


def get_registered_parameters() -> tuple[str, ...]:
    return tuple(FITZHUGH_NAGUMO_MODEL)


def make_model_from_name(name: str) -> sym.Model:
    return FITZHUGH_NAGUMO_MODEL[name]


# }}}
