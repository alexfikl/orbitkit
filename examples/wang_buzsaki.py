# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import replace

import numpy as np

from orbitkit.codegen.numpy import NumpyTarget
from orbitkit.models.symbolic import MatrixSymbol
from orbitkit.models.wang_buzsaki import make_model_from_name
from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

# {{{ create right-hand side

# NOTE: create a nxn all-to-all adjacency matrix for the model
n = 100
A = np.ones(n, dtype=np.int32) - np.eye(n, dtype=np.int32)

figname = "Figure3c"
model = replace(make_model_from_name(f"WangBuzsaki1996{figname}"), A=A)

log.info("Model: %s", type(model))
log.info("Size:  %d", model.n)
log.info("Equations:\n%s", model)

target = NumpyTarget()
source = target.lambdify_model(model, model.n)

# NOTE: lambdify rate functions to get the steady state for the gating variables
V = MatrixSymbol("V", (n,))
hinf = target.lambdify(target.generate_code((V,), model.hinf(V)))
ninf = target.lambdify(target.generate_code((V,), model.ninf(V)))
sinf = target.lambdify(target.generate_code((V,), model.sinf(V)))

# }}}


# {{{ simulation parameters

I_app = {"Figure3a": 1.0, "Figure3b": 1.2, "Figure3c": 1.4}[figname]


def wang_buzsaki_source(t: float, y: Array) -> Array:
    dy = source(t, y)
    dy[: model.n] += I_app / model.param.C

    return dy


tspan = (0.0, 500.0)
tmin_for_plot = 300.0
ylim = (-80.0, 40.0)

V0 = rng.uniform(-70.0, -50.0, size=model.n)
h0 = hinf(V0)
n0 = ninf(V0)
s0 = sinf(V0)

y0 = np.hstack([V0, h0, n0, s0])

log.info("tspan: %s", tspan)
log.info("V0:    %s", V0)
log.info("h0:    %s", h0)
log.info("n0:    %s", n0)
log.info("s0:    %s", s0)
log.info("param: %s", figname)
log.info(model.param)

# }}}


# {{{ evolve

from scipy.integrate import solve_ivp

if n > 32:
    log.warning("Running with %d neurons will be quite slow. Patience..", n)

result = solve_ivp(
    wang_buzsaki_source,
    tspan,
    y0,
    # NOTE: the paper mentions using an RK4 method with dt=0.05ms
    method="RK45",
    max_step=0.05,
)

# }}}

# {{{ plot

try:
    import matplotlib.pyplot as mp  # noqa: F401
except ImportError:
    raise SystemExit(0) from None

from orbitkit.visualization import figure, rastergram, set_plotting_defaults

dirname = pathlib.Path(__file__).parent
set_plotting_defaults()

with figure(dirname / f"wang_buzsaki_{figname.lower()}", overwrite=True) as fig:
    ax = fig.gca()

    mask = result.t > tmin_for_plot
    ax.plot(result.t[mask], result.y[: model.n, mask].T)
    ax.axhline(model.param.V_threshold, color="k", ls="--")
    ax.axhline(model.param.E_syn, color="k", ls=":")

    ax.set_xlabel("$t$ (ms)")
    ax.set_ylabel("$V$ (mV)")
    ax.set_xlim(tmin_for_plot, tspan[1])
    ax.set_ylim(ylim)


with figure(
    dirname / f"wang_buzsaki_{figname.lower()}_rastergram", overwrite=True
) as fig:
    ax = fig.gca()

    rastergram(
        ax,
        result.t,
        result.y[: model.n, :],
        height=0.95 * model.param.V_threshold,
    )

    ax.grid(visible=False, which="both")
    ax.set_xlabel("$t$ (ms)")
    ax.set_ylabel("Neuron")
    ax.set_xlim(tspan[0], tspan[1])
    ax.set_ylim((-0.5, model.n - 0.5))

# }}}
