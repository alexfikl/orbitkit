# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib

import numpy as np

from orbitkit.adjacency import ADJACENCY_TYPES
from orbitkit.utils import module_logger

log = module_logger(__name__)


def main(
    name: str,
    n: int,
    *,
    k: int | None = None,
    seed: int | None = None,
    outfile: pathlib.Path | None,
    force: bool = False,
) -> int:
    if outfile is None:
        outfile = pathlib.Path(f"visualize-adjacency-{name}")

    if not force and outfile.exists():
        log.error("File already exists (use --force to overwrite): '%s'.", outfile)
        return 1

    from orbitkit.adjacency import make_adjacency_matrix_from_name, stringify_adjacency

    rng = np.random.default_rng(seed=seed)
    mat = make_adjacency_matrix_from_name(n, name, k=k, rng=rng)
    log.info("Adjacency matrix:\n%s", stringify_adjacency(mat))

    import networkx as nx  # ty: ignore[unresolved-import,unused-ignore-comment]

    graph = nx.from_numpy_array(mat)
    if name in {"ring", "ring1", "ring2", "strogatzwatts", "startree"}:
        layout = nx.circular_layout(graph)
    else:
        layout = nx.spring_layout(graph, iterations=1024)

    import matplotlib.pyplot as mp

    from orbitkit.visualization import figure, set_plotting_defaults

    set_plotting_defaults()
    with figure(outfile) as fig:
        ax = fig.gca()

        degrees = [graph.degree(n) for n in graph.nodes()]
        nx.draw_networkx_nodes(
            graph,
            layout,
            node_color=degrees,
            cmap=mp.cm.Reds,  # ty: ignore[unresolved-attribute]
            ax=ax,
        )
        nx.draw_networkx_labels(graph, layout, ax=ax)
        nx.draw_networkx_edges(
            graph,
            layout,
            ax=ax,
            arrows=True,
            connectionstyle="arc3,rad=0.1",
        )
        ax.set_axis_off()

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        choices=list(ADJACENCY_TYPES),
        default="ring",
        help="Name of the adjacency matrix type",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Name of the output file",
    )
    parser.add_argument("-s", "--size", type=int, default=16)
    parser.add_argument("-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    raise SystemExit(
        main(
            args.name,
            args.size,
            k=args.k,
            seed=args.seed,
            outfile=args.outfile,
            force=args.force,
        )
    )
