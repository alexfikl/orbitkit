# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.codegen.numpy import NumpyTarget
from orbitkit.models.astrocyte import make_model_from_name
from orbitkit.utils import module_logger, on_ci

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

# {{{ create right-hand side

figname = "Figure3"
model = make_model_from_name(f"LiRinzel1994{figname}")

log.info("Model: %s", type(model))
log.info("Equations:\n%s", model)

target = NumpyTarget()
source = target.lambdify_model(model, 1)

# }}}

# {{{ evolve

tspan = (0.0, 100.0)
y0 = np.array([0.2, 0.5])

from scipy.integrate import solve_ivp

result = solve_ivp(
    source,
    tspan,
    y0,
    method="RK45",
    atol=1.0e-8,
    rtol=1.0e-10,
    max_step=0.1,
)

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

with figure(
    dirname / f"li_rinzel_{figname.lower()}",
    figsize=(10, 5),
    overwrite=True,
) as fig:
    ax = fig.gca()

    ax.plot(result.t, result.y[0], label=r"$C$ (microM)")
    ax.plot(result.t, result.y[1], label="$h$")

    ax.set_xlabel("$t$ (s)")
    ax.set_xlim(tspan[0], tspan[1])
    ax.set_ylim((0.0, 0.8))
    ax.legend()

# }}}
