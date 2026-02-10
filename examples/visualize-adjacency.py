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
        # NOTE: writing GEXF by default because that requires no external dependencies
        outfile = pathlib.Path(f"visualize-adjacency-{name}.gexf")

    if not force and outfile.exists():
        log.error("File already exists (use --force to overwrite): '%s'.", outfile)
        return 1

    from orbitkit.adjacency import make_adjacency_matrix_from_name, stringify_adjacency

    rng = np.random.default_rng(seed=seed)
    mat = make_adjacency_matrix_from_name(n, name, k=k, rng=rng)
    log.info("Adjacency matrix:\n%s", stringify_adjacency(mat))

    if outfile.suffix == ".gexf":
        from orbitkit.visualization import write_gexf_from_adjacency

        write_gexf_from_adjacency(outfile, mat, overwrite=force)
    elif outfile.suffix == ".dot":
        from orbitkit.visualization import write_dot_from_adjacency

        write_dot_from_adjacency(outfile, mat, overwrite=force)
    else:
        from orbitkit.visualization import (
            NetworkXLayout,
            set_plotting_defaults,
            write_nx_from_adjacency,
        )

        if name in {"ring", "ring1", "ring2", "strogatzwatts", "startree"}:
            layout = NetworkXLayout.Circular
        else:
            layout = NetworkXLayout.ARF

        set_plotting_defaults()
        write_nx_from_adjacency(outfile, mat, layout=layout, overwrite=force)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        choices=list(ADJACENCY_TYPES),
        nargs="?",
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
