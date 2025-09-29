# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.models.kuramoto import make_model_from_name
from orbitkit.utils import module_logger

log = module_logger(__name__)
rng = np.random.default_rng(seed=45)

# {{{ create right-hand side

# NOTE: this seems to be quite dependent on the random initial conditions, so
# sometimes it syncs and sometimes there's a chimera -- need to try!
n = 64
figname = "Figure2a"
model = make_model_from_name(f"Abrams2008{figname}")
source = model.lambdify((n, n))

log.info("Model: %s", type(model))
log.info("Size:  %d", 2 * n)
for i, eq in enumerate(model.pretty(use_unicode=True)):
    log.info("Eq%d:\n%s", i, eq)

# }}}


# {{{ simulation parameters

tspan = (0.0, 2000.0)
tmin_for_plot = 0.0

# NOTE: Figure2 says that the integration began from an initial condition close
# to the chimera state. From Figure 1, it seems like this means something like this
y0 = np.hstack([
    rng.normal(0.0, 0.1, size=n),
    rng.normal(0.0, 2.0, size=n),
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
    # atol=1.0e-6,
    # rtol=1.0e-8,
    max_step=0.05,
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
# FIXME: the order parameter does not seem to be as smooth as in Figure 2, not
# quite sure what that's about, but the general trends seem to be the same.
mask = result.t > tmin_for_plot

with figure(
    dirname / f"abrams_kuramoto_{figname.lower()}_order",
    figsize=(10, 10),
    nrows=2,
    ncols=1,
    overwrite=True,
) as fig:
    ax0, ax1 = fig.axes

    theta = result.y[:n, mask]
    r = np.abs(np.mean(np.exp(1j * theta), axis=0))

    ax0.plot(result.t[mask], r, lw=2)
    ax0.set_xlabel("$t$")
    ax0.set_ylabel("$r^0$")
    ax0.set_xlim(tmin_for_plot, tspan[1])
    ax0.set_ylim(0.0, 1.0)

    theta = result.y[n:, mask]
    r = np.abs(np.mean(np.exp(1j * theta), axis=0))

    ax1.plot(result.t[mask], r, lw=2)
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$r^1$")
    ax1.set_xlim(tmin_for_plot, tspan[1])
    ax1.set_ylim(0.0, 1.0)

with figure(
    dirname / f"abrams_kuramoto_{figname.lower()}_final",
    figsize=(10, 10),
    nrows=1,
    ncols=2,
    overwrite=True,
) as fig:
    ax0, ax1 = fig.axes

    theta = model.shift(result.y[:n, -1])
    ax0.plot(np.arange(n), theta, "o")
    ax0.set_xlabel("$j$")
    ax0.set_ylabel(r"$\theta^0_j$")
    ax0.set_ylim([-np.pi, np.pi])

    theta = model.shift(result.y[n:, -1])
    ax1.plot(np.arange(n, 2 * n), theta, "o")
    ax1.set_xlabel("$j$")
    ax1.set_ylabel(r"$\theta^1_j$")
    ax1.set_ylim([-np.pi, np.pi])

# }}}
