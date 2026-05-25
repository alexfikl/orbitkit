# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.adjacency import generate_adjacency_astrocyte_lattice
from orbitkit.models.astrocyte import make_lallouette_mesh
from orbitkit.utils import module_logger, on_ci

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)


# {{{ topology

dim = 2
n = 7
a = 1.0
variant = "scale-free"

points = make_lallouette_mesh(n, dim, a_std=0.35, rng=rng)
mat = generate_adjacency_astrocyte_lattice(
    points,
    variant=variant,
    k_nearest_neighbors=4 if variant == "regular-degree" else 1,
    max_neighbor_distance=1.5 * a,
    rc=0.5 * a,
    rng=rng,
)

# }}}

# {{{ plot

if on_ci():
    raise SystemExit(0)

try:
    import matplotlib.pyplot as mp  # noqa: F401
except ImportError:
    raise SystemExit(0) from None

from orbitkit.visualization import (
    figure,
    set_plotting_defaults,
    write_graph_with_positions,
)

dirname = pathlib.Path(__file__).parent
set_plotting_defaults()

with figure(
    dirname / f"lallouette_network_{variant.lower()}".replace("-", "_"),
    overwrite=True,
) as fig:
    ax = fig.gca()

    write_graph_with_positions(ax, points, mat)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(rf"\texttt{{{variant}}}")

# }}}
