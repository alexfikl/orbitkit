# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import replace

import numpy as np

from orbitkit.models.kuramoto import make_model_from_name
from orbitkit.models.symbolic import stringify
from orbitkit.models.targets import NumpyTarget
from orbitkit.utils import module_logger

log = module_logger(__name__)

# {{{ create right-hand side

figname = "Figure1b"
model = make_model_from_name(f"Schroder2017{figname}")
model = replace(model, K=(model.K != 0).astype(np.uint8))

log.info("Model: %s", type(model))
log.info("Size:  %d", model.n)

args, exprs = model.symbolify(model.n, full=True)
for i, (name, eq) in enumerate(zip(args[1:], exprs, strict=True)):
    log.info("Eq%d:\n    d%s/dt = %s", i, stringify(name), stringify(eq))

# }}}

# {{{ evolve

from scipy.integrate import solve_ivp

tspan = (0.0, 40.0)
tmin_for_plot = 0.0

log.info("tspan: %s", tspan)
log.info("param: %s", figname)
log.info(model)

K = np.linspace(0.0, 1.5, 256)
r_net = np.empty_like(K)
r_link = np.empty_like(K)
r_mf = np.empty_like(K)
r_uni = np.empty_like(K)

for i in range(K.size):
    log.info("Running for K = %.2f", K[i])
    model_i = replace(model, K=K[i] * model.K)

    target = NumpyTarget()
    source = target.lambdify_model(model_i, model_i.n)

    y0 = np.zeros(model_i.n)
    result = solve_ivp(
        source,
        tspan,
        y0,
        method="RK45",
        atol=1.0e-6,
        rtol=1.0e-8,
        max_step=0.01,
    )

    import orbitkit.synchrony as sync

    r_net[i] = sync.global_kuramoto_order_parameter_network(result.y, model.K)
    r_link[i] = sync.global_kuramoto_order_parameter_link(result.y, model.K)
    r_mf[i] = sync.global_kuramoto_order_parameter_mean_field(result.y, model.K)
    r_uni[i] = sync.global_kuramoto_order_parameter_universal(result.y, model.K)

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

with figure(
    dirname / "kuramoto_order_parameter",
    figsize=(15, 10),
    nrows=2,
    ncols=2,
    overwrite=True,
) as fig:
    ax = fig.axes

    from scipy.signal import find_peaks

    dr_uni = np.diff(r_uni) / (K[1] - K[0])
    peaks, _ = find_peaks(dr_uni)

    for i, (name, r) in enumerate([
        ("net", r_net),
        ("link", r_link),
        ("mf", r_mf),
        ("uni", r_uni),
    ]):
        for p in peaks:
            ax[i].axvline(K[p], linestyle="--", color="k", alpha=0.5)

        ax[i].plot(K, r, "o", ms=5)
        ax[i].set_xlabel("$K$")
        ax[i].set_ylabel("$r$")
        ax[i].set_xlim(0.0, 1.5)
        ax[i].set_ylim(0.0, 1.0)
        ax[i].set_title(rf"$r_{{\text{{{name}}}}}$")

# }}}
