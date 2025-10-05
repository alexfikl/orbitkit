# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.models.fitzhugh_nagumo import (
    FitzHughNagumoOmelchenko,
    make_model_from_name,
)
from orbitkit.models.symbolic import stringify
from orbitkit.models.targets import NumpyTarget
from orbitkit.utils import module_logger

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

# {{{ create right-hand side

figname = "Figure1"
model = make_model_from_name(f"Omelchenko2019{figname}")
assert isinstance(model, FitzHughNagumoOmelchenko)

log.info("Model: %s", type(model))
log.info("Size:  %d", model.n)

args, exprs = model.symbolify(model.n, full=True)
for i, (name, eq) in enumerate(zip(args[1:], exprs, strict=True)):
    log.info("Eq%d:\n    d%s/dt = %s", i, stringify(name), stringify(eq))

target = NumpyTarget()
source = target.lambdify_model(model, model.n)

# }}}


# {{{ evolve

from scipy.integrate import solve_ivp

tspan = (0.0, 120.0)
tmin_for_plot = 112.0

log.info("tspan: %s", tspan)
log.info("param: %s", figname)
log.info(model)

# NOTE: Figure1 just says that the initial conditions are "random". We use the
# [-2.0, 2.0] interval to match the magnitude of he solutions at T
y0 = np.hstack([
    rng.uniform(-2.0, 2.0, size=model.n),
    rng.uniform(-2.0, 2.0, size=model.n),
])

result = solve_ivp(
    source,
    tspan,
    y0,
    method="RK45",
    # atol=1.0e-6,
    # rtol=1.0e-8,
    # max_step=0.05,
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

mask = result.t > tmin_for_plot

with figure(
    dirname / f"fitzhugh_nagumo_omelchenko_{figname.lower()}_solution",
    figsize=(10, 3),
    overwrite=True,
) as fig:
    ax = fig.gca()

    n = np.arange(model.n)
    t = result.t[mask]
    t, n = np.meshgrid(t, n)
    im = ax.contourf(n, t, result.y[: model.n, mask], cmap="jet")

    ax.set_xlabel("$i$")
    ax.set_ylabel("$t$")
    # ax.set_xlim(0, model.n)
    # ax.set_ylim(tmin_for_plot, tspan[-1])
    fig.colorbar(im, ax=ax)

with figure(
    dirname / f"fitzhugh_nagumo_omelchenko_{figname.lower()}_final",
    figsize=(10, 3),
    overwrite=True,
) as fig:
    ax = fig.gca()

    ax.plot(np.arange(model.n), result.y[: model.n, -1], "o", markersize=5)
    ax.set_xlabel("$i$")
    ax.set_ylabel("$u_i$")

# with figure(
#     dirname / f"fitzhugh_nagumo_omelchenko_{figname.lower}_phase_velocity",
#     overwrite=True,
# ) as fig:
#     ax = fig.gca()

# }}}
