# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.codegen.numpy import NumpyTarget
from orbitkit.models.kuramoto import Kuramoto
from orbitkit.utils import module_logger, on_ci

log = module_logger(__name__)

try:
    import leidenalg  # ruff: ignore[unused-import]
except ImportError:
    log.info("clustering functionality requires 'leidenalg'")
    raise SystemExit(0) from None

rng = np.random.default_rng(42)

# {{{ parameter setup

n_communities = 3
nodes_per_group = 8
n = n_communities * nodes_per_group

# ground-truth community labels
truth = np.repeat(np.arange(n_communities), nodes_per_group)

# coupling matrix: positive within-community, negative between
K = np.zeros((n, n))
for c in range(n_communities):
    a, b = c * nodes_per_group, (c + 1) * nodes_per_group
    K[a:b, a:b] = 10.0

for c1 in range(n_communities):
    s1 = slice(c1 * nodes_per_group, (c1 + 1) * nodes_per_group)
    for c2 in range(c1 + 1, n_communities):
        s2 = slice(c2 * nodes_per_group, (c2 + 1) * nodes_per_group)
        K[s1, s2] = K[s2, s1] = -2.0

np.fill_diagonal(K, 0.0)

# forcing: avoids obvious synchronization
omega = rng.normal(0.0, 0.5, n)
alpha = 0.0

# }}}

# {{{ evolve

target = NumpyTarget()
model = Kuramoto(omega=omega, alpha=alpha, K=K)
rhs = target.lambdify_model(model, n)

from scipy.integrate import solve_ivp

tspan = (0.0, 300.0)
y0 = rng.uniform(0.0, 2.0 * np.pi, n)

log.info("Integrating %d oscillators over tspan = %s", n, tspan)
result = solve_ivp(rhs, tspan, y0, method="RK45", atol=1e-6, rtol=1e-8, max_step=0.01)

log.info("Integration complete: %d time points", result.t.size)

# }}}

# {{{ build signed correlation matrix

from orbitkit.clusters import make_spearman_weight_matrix

cutoff = int(0.2 * result.t.size)
mat = make_spearman_weight_matrix(np.cos(result.y[:, cutoff:]))

log.info(
    "Correlation matrix: mean %.3f range [%.3f, %.3f]",
    np.mean(mat),
    np.min(mat),
    np.max(mat),
)

# }}}

# {{{ community detection

from orbitkit.clusters import signed_leiden_communities

# NOTE: resolution < 3.0: works nicely
#       resolution~3.25: this seems to break a bit and give more communities
#       resolution>4.00: just gives independent communities for each node
communities = signed_leiden_communities(mat, resolution=1.0, seed=42)

log.info("")
log.info("Ground-truth communities:")
truth_sets = []
for c in range(n_communities):
    nodes = np.where(truth == c)[0]
    truth_sets.append(set(nodes))
    log.info("  G%d: %s", c, nodes)

log.info("")
log.info("Detected communities:")
for cid, comm in enumerate(communities):
    nodes = sorted(comm)
    log.info("  C%d: %s", cid, nodes)

log.info("")
log.info("Recovery per ground-truth group:")
n_correct = 0
available = set(range(n_communities))

for cid, comm in enumerate(communities):
    if not available:
        break

    best = max(available, key=lambda c: len(comm & truth_sets[c]))
    overlap = len(comm & truth_sets[best])
    n_correct += overlap
    available.discard(best)
    log.info("  C%d > G%d: %d / %d correct", cid, best, overlap, len(truth_sets[best]))

for c in sorted(available):
    log.info("  ?? > G%d: 0 / %d correct ", c, len(truth_sets[c]))

log.info("Total: %d / %d nodes", n_correct, n)

# }}}

# {{{ plot

if on_ci():
    raise SystemExit(0)

try:
    import matplotlib.pyplot as mp  # ruff:ignore[unused-import]
except ImportError:
    raise SystemExit(0) from None

from orbitkit.visualization import (
    figure,
    get_color_cycle,
    make_colorbar_axes,
    set_plotting_defaults,
)

dirname = pathlib.Path(__file__).parent
set_plotting_defaults()

# sort nodes by detected community for clean block structure
sorted_idx = np.array([node for comm in communities for node in sorted(comm)])
corr = mat[np.ix_(sorted_idx, sorted_idx)]

# community boundary positions (between communities)
boundaries = np.cumsum([len(c) for c in communities])[:-1] - 0.5
colors = get_color_cycle()

with figure(
    dirname / "kuramoto_signed_communities", figsize=(15, 7), overwrite=True
) as fig:
    fig.delaxes(fig.axes[0])

    # correlation heatmap
    ax = fig.add_subplot(1, 2, 1)
    im = ax.imshow(corr, cmap="RdBu", vmin=-1.0, vmax=1.0, origin="upper")
    fig.colorbar(im, cax=make_colorbar_axes(ax))

    ax.set_box_aspect(1)
    ax.grid(visible=False, which="both")
    ax.tick_params(which="minor", length=0)

    if len(boundaries) > 0:
        for pos in boundaries:
            ax.axhline(pos, color="black", linewidth=4.0)
            ax.axvline(pos, color="black", linewidth=4.0)

    ax.set_title("Spearman Correlation")
    ax.set_xlabel("Node")
    ax.set_ylabel("Node")

    # polar snapshot at final time
    ax = fig.add_subplot(1, 2, 2, polar=True)
    theta = result.y[:, -1]
    for cid, comm in enumerate(communities):
        color = colors[cid % len(colors)]
        for node in sorted(comm):
            ax.plot(
                [0, theta[node]],
                [0, 1],
                color=color,
                linewidth=1.5,
                alpha=0.8,
            )

    ax.set_title("Final Phases")
    ax.set_yticklabels([])

# }}}
