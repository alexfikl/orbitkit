# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import replace

import numpy as np
import sympy as sp

from orbitkit.adjacency import generate_adjacency_erdos_renyi
from orbitkit.models.pfeuty import make_model_from_name
from orbitkit.utils import module_logger

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

# {{{ right-hand side

# NOTE: create a nxn all-to-all adjacency matrix for the model
n = 200
Ainh = generate_adjacency_erdos_renyi(n, k=50, symmetric=False, rng=rng)
Agap = generate_adjacency_erdos_renyi(n, k=10, symmetric=True, rng=rng)

figname = "Figure2cl"
model = replace(make_model_from_name(f"Pfeuty2007{figname}"), Ainh=Ainh, Agap=Agap)

log.info("Model: %s", type(model))
log.info("Size:  %d", model.n)
for i, eq in enumerate(model.pretty()):
    log.info("Eq%d:\n%s", i, eq)

# NOTE: lambdify rate functions to get the steady state for the gating variables
V = sp.var("V")
hinf = sp.lambdify((V,), model.hinf(V), modules="numpy")
ninf = sp.lambdify((V,), model.ninf(V), modules="numpy")

# }}}


# {{{ simulation parameters

tspan = (0.0, 1500.0)
tmin_for_plot = 1000.0
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


# {{{ numbafy

import numba

from orbitkit.models.symbolic import make_sym_vector
from orbitkit.typing import Array

# jit compile lower level function
t = sp.Symbol("t")
args = [make_sym_vector(name, model.n) for name in model.variables]
expr = model.evaluate(t, *args)

source = sp.lambdify((t, *args), expr, modules="numpy")
source = numba.njit(source)

# set up exterior forcing current
# NOTE: this depends on the number of neurons apparently, so it won't give the
# same frequency if n is changed.
C = model.param.C
I_ext = {"Figure2cl": 4.5, "Figure2cr": 1.6}[figname]

# set up noise current
# NOTE: this is meant to act like some form of white noise. because we use an
# adaptive RK method, we can't just add Gaussian to the right-hand side, but
# instead we make a piecewise constant current with proper scaling that gets
# added into the equaion -- this should have the current stochastic behavior
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


@numba.njit  # type: ignore[misc]
def source_with_noise(t: float, y: Array, I_noise: Array) -> Array:
    Vs_i = y[:n]
    Vd_i = y[n : 2 * n]
    h_i = y[2 * n : 3 * n]
    n_i = y[3 * n : 4 * n]
    s_i = y[4 * n :]

    dy = source(t, Vs_i, Vd_i, h_i, n_i, s_i)

    dy[:n] += I_noise / C
    dy[n : 2 * n] += I_ext / C

    return dy  # type: ignore[no-any-return]


def pfeuty_source(t: float, y: Array) -> Array:
    k = min(int(np.digitize(t, noise_t)), noise_t.size - 1)
    return source_with_noise(t, y, I_noise_bins[:, k])  # type: ignore[no-any-return]


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
    max_step=0.01,
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

from orbitkit.visualization import figure, rastergram

with figure(dirname / f"pfeuty_{figname.lower()}", overwrite=True) as fig:
    ax = fig.gca()

    mask = result.t > tmin_for_plot
    ax.plot(result.t[mask], result.y[0, mask].T)
    ax.axhline(model.param.V_threshold, color="k", ls="--")
    ax.axhline(model.param.V_inh, color="k", ls=":")

    ax.set_xlabel("$t$ (ms)")
    ax.set_ylabel("$V$ (mV)")
    ax.set_xlim(tmin_for_plot, tspan[1])
    ax.set_ylim(ylim)


with figure(dirname / f"pfeuty_{figname.lower()}_average", overwrite=True) as fig:
    ax = fig.gca()

    mask = result.t > tmin_for_plot
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
