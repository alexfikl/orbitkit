# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import replace

import numpy as np

from orbitkit.codegen.numpy import NumpyTarget
from orbitkit.models.kuramoto import Kuramoto, make_model_from_name
from orbitkit.utils import module_logger, on_ci

log = module_logger(__name__)

# {{{ create right-hand side

figname = "Figure1b"
model = make_model_from_name(f"Schroder2017{figname}")

assert isinstance(model, Kuramoto)
assert isinstance(model.K, np.ndarray)
A = (model.K != 0).astype(np.uint8)

log.info("Model: %s", type(model))
log.info("Size:  %d", model.n)
log.info("Equations:\n%s", model)

# }}}

# {{{ evolve

from scipy.integrate import solve_ivp

# NOTE: [Schroder2017] does not mention the exact time step / time horizon they
# use, but this seems to match pretty well with the results from Figure 2
dt = 0.1
tspan = (0.0, 1000.0)

log.info("tspan: %s", tspan)
log.info("param: %s", figname)
log.info(model)

K = np.linspace(0.0, 1.5, 128)
t = np.linspace(tspan[0], tspan[1], int((tspan[1] - tspan[0]) / dt))

r_net = np.empty_like(K)
r_link = np.empty_like(K)
r_mf = np.empty_like(K)
r_uni = np.empty_like(K)

for i in range(K.size):
    log.info("Running for K = %.2f", K[i])
    model_i = replace(model, K=K[i] * A)

    target = NumpyTarget()
    source = target.lambdify_model(model_i, model_i.n)

    y0 = np.zeros(model_i.n)
    result = solve_ivp(
        source,
        tspan,
        y0,
        method="RK45",
        t_eval=t,
        atol=1.0e-6,
        rtol=1.0e-8,
    )

    import orbitkit.synchrony as sync

    theta = result.y
    r_net[i] = sync.global_kuramoto_order_parameter_network(theta, A)
    r_link[i] = sync.global_kuramoto_order_parameter_link(theta, A)
    r_mf[i] = sync.global_kuramoto_order_parameter_mean_field(theta, A)
    r_uni[i] = sync.global_kuramoto_order_parameter_universal(theta, A)

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

    for i, (rname, r) in enumerate([
        ("net", r_net),
        ("link", r_link),
        ("mf", r_mf),
        ("uni", r_uni),
    ]):
        for p in peaks:
            ax[i].axvline(K[p], linestyle="--", color="k", alpha=0.4, lw=1.5)

        ax[i].plot(K, r, "o", ms=5)
        ax[i].set_xlabel("$K$")
        ax[i].set_ylabel("$r$")
        ax[i].set_xlim(0.0, 1.5)
        ax[i].set_ylim(0.0, 1.0)
        ax[i].set_title(rf"$r_{{\text{{{rname}}}}}$")

# }}}
