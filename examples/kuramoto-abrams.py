# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.codegen.jitcode import JiTCODETarget, make_input_variable
from orbitkit.models.kuramoto import make_model_from_name, shift_kuramoto_angle
from orbitkit.utils import module_logger, on_ci, tictoc

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

# FIXME: the order parameter does not seem to be as smooth as in Figure 2, not
# quite sure what that's about, but the general trends seem to be the same.

# FIXME: this seems to be quite dependent on the random initial conditions, so
# sometimes it syncs and sometimes there's a chimera -- need to try!

# {{{ create right-hand side

n = 64
figname = "Figure2c"
model = make_model_from_name(f"Abrams2008{figname}")

log.info("Model: %s", type(model))
log.info("Size:  %d", 2 * n)
log.info("Equations:\n%s", model)

with tictoc("codegen"):
    target = JiTCODETarget()
    source_func = target.lambdify_model(model, (n, n))

if on_ci():
    log.warning("Example is very slow. Quitting..")
    raise SystemExit(0)

# }}}


# {{{ evolve

import jitcode

tspan = (0.0, 1000.0)
tmin_for_plot = 0.0

log.info("tspan: %s", tspan)
log.info("param: %s", figname)
log.info(model)

# NOTE: Figure2 says that the integration began from an initial condition close
# to the chimera state. From Figure 1, it seems like this means something like this
y0 = np.hstack([
    rng.normal(0.0, 0.1, size=n),
    rng.normal(0.0, 2.0, size=n),
])

with tictoc("evaluation"):
    y = make_input_variable((n, n))
    source = source_func(jitcode.t, y)

with tictoc("compile"):
    ode = target.compile(source, y, method="RK45")
    ode.set_initial_value(y0, tspan[0])

with tictoc("evolve"):
    dt = 0.1
    t = np.arange(tspan[0], tspan[1], dt)
    y = np.empty((y0.size, t.size), dtype=y0.dtype)
    for i in range(t.size):
        y[:, i] = ode.integrate(t[i])

# }}}

# {{{ plot

try:
    import matplotlib.pyplot as mp  # noqa: F401
except ImportError:
    raise SystemExit(0) from None

from orbitkit.visualization import figure, set_plotting_defaults

dirname = pathlib.Path(__file__).parent
set_plotting_defaults()

mask = t > tmin_for_plot

with figure(
    dirname / f"abrams_kuramoto_{figname.lower()}_order",
    figsize=(10, 10),
    nrows=2,
    ncols=1,
    overwrite=True,
) as fig:
    ax0, ax1 = fig.axes

    # compute Kuramoto order parameter
    theta = y[:n, mask]
    r0 = np.abs(np.mean(np.exp(1j * theta), axis=0))

    theta = y[n:, mask]
    r1 = np.abs(np.mean(np.exp(1j * theta), axis=0))

    ax0.plot(t[mask], r0, lw=3)
    ax0.set_xlabel("$t$")
    ax0.set_ylabel("$r^0$")
    ax0.set_xlim(tmin_for_plot, tspan[1])
    ax0.set_ylim(0.0, 1.0)

    ax1.plot(t[mask], r1, lw=3)
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

    theta = shift_kuramoto_angle(y[:n, -1])
    ax0.plot(np.arange(n), theta, "o")
    ax0.set_xlabel("$j$")
    ax0.set_ylabel(r"$\theta^0_j$")
    ax0.set_ylim([-np.pi, np.pi])

    theta = shift_kuramoto_angle(y[n:, -1])
    ax1.plot(np.arange(n, 2 * n), theta, "o")
    ax1.set_xlabel("$j$")
    ax1.set_ylabel(r"$\theta^1_j$")
    ax1.set_ylim([-np.pi, np.pi])

# }}}
