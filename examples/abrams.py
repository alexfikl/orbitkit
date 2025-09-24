# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.models.kuramoto import make_model_from_name
from orbitkit.utils import module_logger

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

# {{{ create right-hand side

n = 64
figname = "Figure2b"
model = make_model_from_name(f"Abrams2008{figname}")
source = model.lambdify((n, n))

log.info("Model: %s", type(model))
log.info("Size:  %d", 2 * n)
for i, eq in enumerate(model.pretty(use_unicode=True)):
    log.info("Eq%d:\n%s", i, eq)

# }}}


# {{{ simulation parameters

tspan = (0.0, 3000.0)
tmin_for_plot = 2000.0

# NOTE: Figure2 says that the integration began from an initial condition close
# to the chimera state. From Figure 1, it seems like this means something like this
y0 = np.hstack([
    np.zeros(n),
    rng.uniform(-np.pi, np.pi, size=n),
])

log.info("tspan: %s", tspan)
log.info("param: %s", figname)
log.info(model)

# }}}


# {{{ evolve

from scipy.integrate import solve_ivp

result = solve_ivp(
    source,
    tspan,
    y0,
    method="RK45",
    # atol=1.0e-8,
    # rtol=1.0e-10,
    max_step=1.0,
)

# }}}

# {{{ plot

try:
    import matplotlib.pyplot as mp  # noqa: F401
except ImportError:
    raise SystemExit(0) from None

from orbitkit.utils import set_plotting_defaults

dirname = pathlib.Path(__file__).parent
set_plotting_defaults()

from orbitkit.visualization import figure

# compute Kuramoto order parameter for the sigma=2 population
mask = result.t > tmin_for_plot
theta = result.y[n:, mask]
r = np.abs(np.mean(np.exp(1j * theta), axis=0))

with figure(
    dirname / f"abrams_kuramoto_{figname.lower()}_order",
    figsize=(10, 5),
    overwrite=True,
) as fig:
    ax = fig.gca()

    ax.plot(result.t[mask], r, lw=2)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$r$")
    ax.set_xlim(tmin_for_plot, tspan[1])
    ax.set_ylim(0.0, 1.0)

# }}}
