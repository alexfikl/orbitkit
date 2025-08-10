# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import sympy as sp

from orbitkit.models.wang_rinzel import make_model_from_name
from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)
rng = np.random.default_rng(seed=None)

# {{{ create right-hand side

model = make_model_from_name("Symbolic")
sym_source, _ = model.symbolize()

log.info("Model (symbolic):")
for expr in sym_source:
    sp.pprint(expr, use_unicode=False)

figname = "Figure4a"
model = make_model_from_name(f"WangRinzel1992{figname}")

sym_model, _ = model.symbolize()
source = model.lambdify()

# NOTE: printing this is not very helpful
# log.info("Eq1: %s", sym_model[0])
# log.info("Eq2: %s", sym_model[1])
# log.info("Eq3: %s", sym_model[2])
# log.info("Eq4: %s", sym_model[3])

# }}}


# {{{ simulation parameters


def figure3a_source(t: float, y: Array) -> Array:
    # NOTE: Figure3a clamps the voltage V0 to some value. We fake that here by
    # removing the right-hand side completely in the interval [380, 580].
    dy = source(t, y)
    if 380.0 < t < 580.0:
        dy[0] = 0.0

    return dy


def figure3c_source(t: float, y: Array) -> Array:
    # NOTE: Figure3c adds a little transient hyperpolarization to kickstart the
    # oscillations. The text is not clear, but this seems to work well enough.
    dy = source(t, y)
    if 200.0 < t < 250.0:
        dy[0] -= 1.0 / model.param.C
        dy[1] -= 1.0 / model.param.C

    return dy


def figure4a_source(t: float, y: Array) -> Array:
    # NOTE: Figure4a has two burst of additional current pumped into the system.
    # This is nicely described in the caption, so should work.
    dy = source(t, y)
    if 300.0 < t < 350.0:
        dy[0] += 1.0 / model.param.C
        dy[1] += 1.0 / model.param.C

    if 1100.0 < t < 1150.0:
        dy[0] += 1.0 / model.param.C
        dy[1] -= 1.0 / model.param.C

    return dy


if figname == "Figure1a":
    tspan = (0.0, 500.0)
    tmin_for_plot = 250.0
    ylim = (-80.0, 0.0)

    # NOTE: V0 is guessed from Figure 1a *very* roughly
    V0 = np.array([-44.0, -67.0])
    h0 = rng.uniform(0.0, 1.0, size=model.n)
    y0 = np.hstack([V0, h0])

    wang_rinzel_source = source
elif figname == "Figure3a":
    tspan = (0.0, 900.0)
    tmin_for_plot = 100.0
    ylim = (-80.0, 20.0)

    # NOTE: V0 is guessed from Figure 3a *very* roughly, h0 seems to also have
    # an impact on how closely the results resemble the paper figure.
    V0 = np.array([-74.0, -33.0])
    h0 = np.array([0.91, 0.91])
    y0 = np.hstack([V0, h0])

    wang_rinzel_source = figure3a_source
elif figname == "Figure3c":
    tspan = (0.0, 500.0)
    tmin_for_plot = 100.0
    ylim = (-80.0, 20.0)

    # NOTE: (V0, h0) is taken from Figure 3c and need to match the fixed point.
    V0 = np.array([-34.3, -50.5])
    h0 = np.array([0.0141, 0.0587])
    y0 = np.hstack([V0, h0])

    wang_rinzel_source = figure3c_source
elif figname == "Figure4a":
    tspan = (0.0, 2000.0)
    tmin_for_plot = 0.0
    ylim = (-90.0, 0.0)

    # NOTE: V0 is guessed from Figured 4a *very* roughly
    V0 = np.array([-58.0, -78.0])
    h0 = rng.uniform(0.0, 1.0, size=model.n)
    s0 = np.array([1.0, 0.0])
    y0 = np.hstack([V0, h0, s0])

    wang_rinzel_source = figure4a_source
else:
    raise ValueError(f"Unknown model parameters: '{figname}'")

log.info("tspan: %s", tspan)
log.info("V0:    %s", V0)
log.info("h0:    %s", h0)
log.info("param: %s", figname)
log.info(model.param)

# }}}


# {{{ evolve

from scipy.integrate import solve_ivp

result = solve_ivp(
    wang_rinzel_source,
    tspan,
    y0,
    method="RK45",
    atol=1.0e-8,
    rtol=1.0e-10,
    max_step=0.1,
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

with figure(
    dirname / f"wang_rinzel_{figname.lower()}_v",
    figsize=(10, 5),
    overwrite=True,
) as fig:
    ax = fig.gca()

    mask = result.t > tmin_for_plot
    ax.plot(result.t[mask], result.y[0, mask], label="$V_1$ (mV)")
    ax.plot(result.t[mask], result.y[1, mask], label="$V_2$ (mV)")
    ax.axhline(model.param.V_threshold, color="k", ls="--")

    ax.set_xlabel("$t$ (ms)")
    ax.set_ylabel("$V$ (mV)")
    ax.set_xlim(tmin_for_plot, tspan[1])
    ax.set_ylim(ylim)
    ax.legend(loc="upper right")

from orbitkit.models.wang_rinzel import WangRinzelExt

if isinstance(model, WangRinzelExt):
    with figure(
        dirname / f"wang_rinzel_{figname.lower()}_s",
        figsize=(10, 2.5),
        overwrite=True,
    ) as fig:
        ax = fig.gca()

        mask = result.t > tmin_for_plot
        ax.plot(result.t[mask], result.y[-2, mask], label="$s_1$")
        ax.plot(result.t[mask], result.y[-1, mask], label="$s_2$")

        ax.set_xlabel("$t$ (ms)")
        ax.set_ylabel("$s$")
        ax.set_xlim(tmin_for_plot, tspan[1])
        ax.set_ylim((0, 1))
        ax.legend(loc="lower left")

# }}}
