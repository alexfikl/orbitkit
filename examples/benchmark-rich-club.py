# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
from rich.table import Table

from orbitkit.adjacency import generate_adjacency_erdos_renyi, rewire_adjacency
from orbitkit.metrics import compute_rich_club_coefficient
from orbitkit.utils import BlockTimer, module_logger, on_ci, stringify_table

log = module_logger(__name__)

if on_ci():
    raise SystemExit(0)

try:
    import networkx as nx  # ty: ignore[unresolved-import,unused-ignore-comment]
except ImportError:
    nx = None

rng = np.random.default_rng(seed=42)

# Erdos-Renyi probability
p = 0.15

# (nnodes, q, nsamples)
configs = [
    (50, 10, 1),
    (50, 100, 1),
    (50, 10, 5),
    (100, 10, 1),
    (100, 100, 1),
    (100, 10, 5),
    (200, 10, 1),
    (200, 50, 1),
    (200, 10, 5),
    (400, 10, 1),
    (400, 50, 1),
]

table = Table(
    "N",
    "E",
    "q",
    "nsamples",
    "Unnorm (ms)",
    "Rewire (ms)",
    "Orbitkit (ms)",
    "NetworkX (ms)",
    title="Rich-Club Normalization Benchmark",
)

mats: dict[int, np.ndarray] = {}
tm = BlockTimer("rcc")

for n, q, n_samples in configs:
    if n not in mats:
        mats[n] = generate_adjacency_erdos_renyi(
            n, p=p, symmetric=True, dtype=np.float64, rng=rng
        )

    mat = mats[n]
    m = np.sum(np.triu(mat > 0, k=1))

    # 1. normalized=False time
    with tm:
        _ = compute_rich_club_coefficient(mat, normalized=False)
    t_unnorm = tm.t_wall * 1000.0

    # 2. single rewire_adjacency time
    with tm:
        _ = rewire_adjacency(mat, q=q, rng=rng)
    t_rewire = tm.t_wall * 1000.0

    # 3. total normalized time
    with tm:
        _ = compute_rich_club_coefficient(
            mat, normalized=True, q=q, n_samples=n_samples, rng=rng
        )
    t_ok = tm.t_wall * 1000.0

    # 4. networkx normalized time
    if nx is not None and n <= 200:
        G = nx.from_numpy_array(mat)

        try:
            with tm:
                _ = nx.rich_club_coefficient(
                    G, normalized=True, Q=q, n_samples=n_samples, seed=42
                )
            t_nx = tm.t_wall * 1000.0
            t_nx_str = f"{t_nx:.2f}"
        except nx.NetworkXError:
            t_nx_str = "failed (div 0)"
    else:
        t_nx_str = "N/A"

    table.add_row(
        str(n),
        str(m),
        str(q),
        str(n_samples),
        f"{t_unnorm:.3f}",
        f"{t_rewire:.2f}",
        f"{t_ok:.2f}",
        t_nx_str,
    )

log.info("\n%s", stringify_table(table))
