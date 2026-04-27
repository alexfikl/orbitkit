# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from itertools import product

import numpy as np

from orbitkit.models.theta import ThetaModel, find_fixed_points
from orbitkit.symbolic.primitives import DiracDelayKernel
from orbitkit.utils import module_logger, on_ci

log = module_logger(__name__)

# {{{ fixed points

npoints = 256
kappa = np.linspace(-5.0, 5.0, npoints)
eta = np.linspace(-1.0, 1.0, npoints)

fp_type1 = np.empty((kappa.size, eta.size), dtype=np.int8)
fp_type2 = np.empty((kappa.size, eta.size), dtype=np.int8)

for i, j in product(range(kappa.size), range(eta.size)):
    model = ThetaModel(kappa=kappa[i], eta=eta[j], h=DiracDelayKernel(tau=0.0))
    fp = find_fixed_points(model)

    fp_type1[i, j] = fp.on_circle.size
    fp_type2[i, j] = fp.in_disk.size

# }}}

# {{{ plot

if on_ci():
    raise SystemExit(0)

try:
    import matplotlib.pyplot as mp  # noqa: F401
except ImportError:
    raise SystemExit(0) from None

from orbitkit.visualization import (
    discrete_colorbar,
    discrete_heatmap,
    figure,
    set_plotting_defaults,
)

dirname = pathlib.Path(__file__).parent
set_plotting_defaults()


def plot_fixed_points(ax, x, y, z, *, title: str) -> None:
    nlevels = int(z.max()) + 1
    im = discrete_heatmap(ax, x, y, z.T, vmin=0, vmax=nlevels - 1)
    discrete_colorbar(im, ax=ax, nlevels=nlevels)

    ax.axhline(0.0, color="k", lw=1)
    ax.axvline(0.0, color="k", lw=1)

    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\eta$")
    ax.set_title(title)


with figure(dirname / "theta_fixed_points", nrows=1, ncols=2, overwrite=True) as fig:
    ax1, ax2 = fig.axes

    plot_fixed_points(ax1, kappa, eta, fp_type1, title="On-Circle Fixed Points")
    plot_fixed_points(ax2, kappa, eta, fp_type2, title="In-Disk Fixed Points")

# }}}
