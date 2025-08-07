# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

import orbitkit.models.symbolic as sym
from orbitkit.models.wang_rinzel import make_model_from_name
from orbitkit.utils import module_logger

log = module_logger(__name__)
rng = np.random.default_rng(seed=None)

# {{{ create right-hand side

model = make_model_from_name("WangRinzel1992Figure1a")

t = sym.make_variable("t")
V = sym.make_sym_vector("V", model.n)
h = sym.make_sym_vector("h", model.n)

sym_source = model(t, V, h)
source = sym.lambdify(sym_source, V, h)
log.info("System:\n%s", sym_source)

# }}}


# {{{ evolve

from scipy.integrate import solve_ivp

tspan = (0.0, 500.0)
V0 = rng.uniform(-70.0, -30.0, size=model.n)
h0 = rng.uniform(0.0, 1.0, size=model.n)

log.info("tspan: %s", tspan)
log.info("V0:    %s", V0)
log.info("h0:    %s", h0)

y0 = np.hstack([V0, h0])
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

try:
    import matplotlib.pyplot as mp  # noqa: F401
except ImportError:
    raise SystemExit(0) from None

from orbitkit.utils import figure, set_plotting_defaults

dirname = pathlib.Path(__file__).parent
set_plotting_defaults()

with figure(dirname / "wang_rinzel", overwrite=True) as fig:
    ax = fig.gca()

    ax.plot(result.t, result.y[0], label="$V_1$ (mV)")
    ax.plot(result.t, result.y[1], label="$V_2$ (mV)")
    ax.axhline(model.param.V_threshold, color="k", ls="--")

    ax.set_xlabel("$t$ (ms)")
    ax.set_ylim(tspan)
    ax.set_ylim((-80.0, 0.0))
    ax.legend()

# }}}
