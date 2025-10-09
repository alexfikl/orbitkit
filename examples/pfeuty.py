# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import replace

import numpy as np

from orbitkit.adjacency import generate_adjacency_erdos_renyi
from orbitkit.codegen.numpy import NumpyTarget
from orbitkit.models.pfeuty import make_model_from_name
from orbitkit.models.symbolic import MatrixSymbol
from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

# FIXME: This does not yet reproduce the results from Figure2C in [Pfeuty2007]_.
# It's not clear what is wrong with the parameters / setup used here, but the
# frequency of the neuron spikes does not seem to match what's in the paper.

# FIXME: [Pfeuty2007]_ does not mention initial conditions. This uses something
# similar to [WangBuzsaki1996]_, but it's not clear if that's a good idea. Might
# be causing the matching issue above?

# {{{ right-hand side

# NOTE: create an nxn random adjacency matrix for the model
n = 200
A_inh = generate_adjacency_erdos_renyi(n, k=50, symmetric=False, rng=rng)
A_gap = generate_adjacency_erdos_renyi(n, k=10, symmetric=True, rng=rng)

figname = "Figure2cl"
model = replace(make_model_from_name(f"Pfeuty2007{figname}"), A_inh=A_inh, A_gap=A_gap)

log.info("Model: %s", type(model))
log.info("Size:  %d", model.n)
log.info("Equations:\n%s", model)

target = NumpyTarget()
source = target.lambdify_model(model, model.n)

# NOTE: lambdify rate functions to get the steady state for the gating variables
V = MatrixSymbol("V", (n,))
hinf = target.lambdify(target.generate_code((V,), model.hinf(V)))
ninf = target.lambdify(target.generate_code((V,), model.ninf(V)))

# }}}


# {{{ simulation parameters

tspan = (0.0, 1500.0)
tmin_for_plot = 1250.0
ylim = (-80.0, 40.0)

Vs0 = model.param.V_L + rng.uniform(-3.0, 3.0, size=n)
Vd0 = model.param.V_L + rng.uniform(-3.0, 3.0, size=n)
h0 = hinf(Vs0)
n0 = ninf(Vs0)
s0 = np.zeros_like(Vd0)

y0 = np.hstack([Vs0, Vd0, h0, n0, s0])

log.info("tspan: %s", tspan)
log.info("param: %s", figname)
log.info(model.param)

# }}}


# {{{ add noise

# set up exterior forcing current
# NOTE: this depends on the number of neurons apparently, so it won't give the
# same frequency if n is changed.
C = model.param.C
I_ext = {"Figure2cl": 15.0, "Figure2cr": 1.6}[figname]

# set up noise current NOTE: this is meant to act like some form of white
# noise. Because we use an adaptive RK method, we can't just add random noise
# to the right-hand side at every time step, but instead we make a piecewise
# constant random current with proper scaling that gets added into the equation
# -- this should have the correct stochastic behavior
noise_mu = 0.0
noise_sigma = {"Figure2cl": 0.1, "Figure2cr": 0.8}[figname]

# NOTE: this is mostly meant to mimic the paper, where they used a second-order
# RK method with a fixed step size of dt = 0.01
noise_dt = 0.01
noise_n = int((tspan[1] - tspan[0]) / noise_dt) + 1
noise_t = np.linspace(tspan[0], tspan[1], noise_n)
noise_dt = noise_t[1] - noise_t[0]

eta = rng.standard_normal((n, noise_t.size))
I_noise_bins = noise_sigma / np.sqrt(noise_dt) * eta


def pfeuty_source(t: float, y: Array) -> Array:
    k = min(int(np.digitize(t, noise_t)), noise_t.size - 1)

    dy = source(t, y)
    dy[:n] += I_noise_bins[:, k] / C
    dy[n : 2 * n] += I_ext / C

    return dy


# }}}


# {{{ evolve

from scipy.integrate import solve_ivp

if n > 32:
    log.warning("Running with %d neurons will be quite slow. Patience..", n)

result = solve_ivp(
    pfeuty_source,
    tspan,
    y0,
    # NOTE: the paper mentions using an RK2 method with dt=0.01ms
    method="RK23",
    max_step=noise_dt,
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

# get a nice mask for plotting
mask = result.t > tmin_for_plot

# find a neuron to plot that seems to have spiked the most
from scipy.signal import find_peaks

k = 0
npeaks = 0
for i in range(n):
    peaks_i, _ = find_peaks(result.y[i, mask], height=0.9 * model.param.V_threshold)
    if len(peaks_i) > npeaks:
        npeaks = len(peaks_i)
        k = i
        log.info("Found neuron %d with %d spikes (?).", i, npeaks)

from orbitkit.visualization import figure, rastergram

with figure(dirname / f"pfeuty_{figname.lower()}", overwrite=True) as fig:
    ax = fig.gca()

    ax.plot(result.t[mask], result.y[k, mask].T)
    ax.axhline(model.param.V_threshold, color="k", ls="--")
    ax.axhline(model.param.V_inh, color="k", ls=":")

    ax.set_xlabel("$t$ (ms)")
    ax.set_ylabel(f"$V_{{{k}}}$ (mV)")
    ax.set_xlim(tmin_for_plot, tspan[1])
    ax.set_ylim(ylim)


with figure(dirname / f"pfeuty_{figname.lower()}_average", overwrite=True) as fig:
    ax = fig.gca()

    ax.plot(result.t[mask], np.mean(result.y[: model.n, mask].T, axis=1))
    ax.axhline(model.param.V_threshold, color="k", ls="--")
    ax.axhline(model.param.V_inh, color="k", ls=":")

    ax.set_xlabel("$t$ (ms)")
    ax.set_ylabel(r"$\bar{V}$ (mV)")  # noqa: RUF027
    ax.set_xlim(tmin_for_plot, tspan[1])
    ax.set_ylim(ylim)


with figure(dirname / f"pfeuty_{figname.lower()}_rastergram", overwrite=True) as fig:
    ax = fig.gca()

    rastergram(
        ax,
        result.t[mask],
        result.y[: model.n, mask],
        height=0.9 * model.param.V_threshold,
        markerheight=1.0,
    )

    ax.grid(visible=False, which="both")
    ax.set_xlabel("$t$ (ms)")
    ax.set_ylabel("Neuron")
    ax.set_xlim(tmin_for_plot, tspan[1])
    ax.set_ylim((-0.5, model.n - 0.5))

# }}}
