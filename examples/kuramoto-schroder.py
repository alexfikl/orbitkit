# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.codegen.numpy import NumpyTarget
from orbitkit.models.kuramoto import Kuramoto, make_model_from_name
from orbitkit.utils import module_logger

log = module_logger(__name__)

# {{{ create right-hand side

figname = "Figure1c"
model = make_model_from_name(f"Schroder2017{figname}")
assert isinstance(model, Kuramoto)

log.info("Model: %s", type(model))
log.info("Size:  %d", model.n)
log.info("Equations:\n%s", model)

target = NumpyTarget()
source = target.lambdify_model(model, model.n)

# }}}

# {{{ evolve

from scipy.integrate import solve_ivp

tspan = (0.0, 20.0)
tmin_for_plot = 0.0

log.info("tspan: %s", tspan)
log.info("param: %s", figname)
log.info(model)

y0 = np.zeros(model.n)
result = solve_ivp(
    source,
    tspan,
    y0,
    method="RK45",
    atol=1.0e-6,
    rtol=1.0e-8,
    max_step=0.01,
)

# }}}

# {{{ plot

try:
    import matplotlib.pyplot as mp  # noqa: F401
except ImportError:
    raise SystemExit(0) from None

from orbitkit.visualization import (
    figure,
    set_plotting_defaults,
    to_color,
    write_dot_from_adjacency,
)

dirname = pathlib.Path(__file__).parent
set_plotting_defaults()

with figure(
    dirname / f"kuramoto_{figname.lower()}_solution", figsize=(10, 5), overwrite=True
) as fig:
    ax = fig.gca()

    ax.plot(result.t, result.y.T)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\theta_i(t)$")
    if figname in {"Figure1b", "Figure1c", "Figure1d"}:
        ax.set_ylim(-10.0, 10.0)
    else:
        ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(tspan)

assert isinstance(model.K, np.ndarray)
assert isinstance(model.omega, np.ndarray)

write_dot_from_adjacency(
    dirname / f"kuramoto_{figname.lower()}.dot",
    model.K,  # ty: ignore[invalid-argument-type]
    nodenames=tuple(f"{omega:g}" for omega in model.omega),
    nodecolors=to_color(model.omega),  # ty: ignore[invalid-argument-type]
    overwrite=True,
)

# }}}
