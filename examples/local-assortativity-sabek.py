# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.utils import module_logger, on_ci

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

# NOTE: This tries to reproduce Figure 2b,d,e from [Sabek2023] for a single
# Weighted Random Graph (WRG).
# FIXME: The histogram from the paper seems to be incorrect: it goes up to 1500
# frequency for a graph with n = 1000 nodes. This shouldn't be possible.

# {{{ generate

n = 1000
omega_bar_target = 0.02
alpha = 1.0
beta = 1.0

from orbitkit.adjacency import generate_weighted_random_graph_garlaschelli
from orbitkit.metrics import (
    compute_average_excess_strength,
    compute_local_assortativity_sabek,
)

W = generate_weighted_random_graph_garlaschelli(n, omega=omega_bar_target, rng=rng)

strength = compute_average_excess_strength(W)
rho_v = compute_local_assortativity_sabek(W, alpha=alpha, beta=beta)

# }}}

# {{{ plot

if on_ci():
    raise SystemExit(0)

try:
    import matplotlib.pyplot as mp  # ruff:ignore[unused-import]
except ImportError:
    raise SystemExit(0) from None

from orbitkit.visualization import figure, set_plotting_defaults

dirname = pathlib.Path(__file__).parent
set_plotting_defaults()

with figure(
    dirname / "local_assortativity_sabek",
    ncols=3,
    figsize=(15, 5),
    overwrite=True,
) as fig:
    ax1, ax2, ax3 = fig.axes

    ax1.plot(rho_v, "ko", ms=4, alpha=0.6)
    ax1.axhline(0, linestyle="-")
    ax1.set_xlabel("Index", fontsize=20)
    ax1.set_ylabel("Generalised vertex assortativeness", fontsize=20)

    ax2.scatter(strength, rho_v, s=10, alpha=0.6, color="black")
    ax2.axhline(0, linestyle="-")
    ax2.set_xlabel("Average excess strength", fontsize=20)
    ax2.set_ylabel("Generalised vertex assortativeness", fontsize=20)

    ax3.hist(rho_v, bins=30, color="gray", edgecolor="black")
    ax3.set_xlabel("Generalised vertex assortativeness", fontsize=20)
    ax3.set_ylabel("Frequency", fontsize=20)

# }}}
