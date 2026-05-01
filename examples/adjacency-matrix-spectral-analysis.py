# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.adjacency import (
    generate_random_equal_row_sum,
    make_adjacency_matrix_from_name,
)
from orbitkit.utils import module_logger, on_ci

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

# {{{ compute spectrum

topology = "lattice"
log.info("Analyzing '%s' topology...", topology)

nrepeats = 32
ns = np.arange(16, 512 + 1, 32)

lambda_gaps = np.empty((ns.size, nrepeats))
participation_ratios = np.empty((ns.size, nrepeats))

for i, n in enumerate(ns):
    for j in range(nrepeats):
        mat = make_adjacency_matrix_from_name(n, topology, k=None, rng=rng)
        W = generate_random_equal_row_sum(mat, alpha=1.0, rng=rng)

        # get sorted eigenvalues and eigenvectors
        # NOTE: we only look at the real part here because we are interested in the
        # dynamical systems view: the real part of an eigenvalue dictates stability.
        # This is not strictly true, depending on hoe the system is actually laid
        # out, but it should give some idea..
        eigvals, eigvecs = np.linalg.eig(W)

        order = np.argsort(eigvals.real)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        assert np.all(np.diff(eigvals.real[::-1]) >= 0), eigvals.real

        # NOTE: eigvals[0] = 1 and eigvecs[:, 0] = 1 for equal row sum matrices.
        # We look at the second eigenvector and the gap to understand the graph
        v2 = eigvecs[:, 1].real / np.linalg.norm(eigvecs[:, 1].real)
        lambda_gaps[i, j] = eigvals[0].real - eigvals[1].real
        participation_ratios[i, j] = np.sum(v2**2) ** 2 / np.sum(v2**4)

# }}}


# {{{ plot

if on_ci():
    raise SystemExit(0)

try:
    import matplotlib.pyplot as mp  # noqa: F401
except ImportError:
    raise SystemExit(0) from None

from orbitkit.visualization import figure, set_plotting_defaults

dirname = pathlib.Path(__file__).parent
set_plotting_defaults()

with figure(dirname / f"adjacency_spec_{topology}_lambda_gap", overwrite=True) as fig:
    ax = fig.gca()

    means = np.mean(lambda_gaps, axis=1)
    stds = np.std(lambda_gaps, ddof=1, axis=1)
    (line,) = ax.plot(ns, means)
    ax.fill_between(ns, means - stds, means + stds, color=line.get_color(), alpha=0.15)
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$\lambda_1 - \lambda_2$")

with figure(dirname / f"adjacency_spec_{topology}_part_ratio", overwrite=True) as fig:
    ax = fig.gca()

    means = np.mean(participation_ratios, axis=1)
    stds = np.std(participation_ratios, ddof=1, axis=1)
    (line,) = ax.plot(ns, means)
    ax.fill_between(ns, means - stds, means + stds, color=line.get_color(), alpha=0.15)
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"Participation Ratio")

# }}}
