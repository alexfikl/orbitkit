# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

import orbitkit.symbolic.primitives as sym
from orbitkit.codegen.numpy import NumpyTarget
from orbitkit.models.astrocyte import make_model_from_name
from orbitkit.utils import module_logger, on_ci

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

# FIXME: this fails to reproduce the results from Figure 12 FM. The paper shows
# 2-3 oscillations that then decay at some fixed point C ~ 0.5 and then decay
# further when the Glutamate burst is done.

# {{{ create right-hand side

figname = "Figure12am"
model = make_model_from_name(f"DePitta2009{figname}")

log.info("Model: %s", type(model))
log.info("Equations:\n%s", model)

target = NumpyTarget()
source = target.lambdify_model(model, 1)

# }}}

# {{{ evolve

tspan = (0.0, 300.0)
y0 = np.array([0.05, 0.9, 0.1])

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

t = sym.Variable("t")
gamma = target.lambdify(target.generate_code((t,), model.gamma(t)))  # ty: ignore[unresolved-attribute]


with figure(
    dirname / f"de_pitta_{figname.lower()}",
    nrows=3,
    figsize=(10, 12),
    overwrite=True,
) as fig:
    ax1, ax2, ax3 = fig.axes

    ax1.plot(result.t, result.y[0])
    ax1.set_ylabel(r"[$\text{Ca}^{2+}$] [$\mu M$]")
    ax1.set_xlim(tspan[0], tspan[1])
    ax1.set_ylim((0.0, 1.2))

    ax2.plot(result.t, result.y[2])
    ax2.set_ylabel(r"[$\text{IP}_3$] [$\mu M$]")
    ax2.set_xlim(tspan[0], tspan[1])
    ax2.set_ylim((0.0, 3))

    ax3.semilogy(result.t, np.vectorize(gamma)(result.t))
    ax3.set_xlabel("$t$ (s)")
    ax3.set_ylabel(r"[Glu] [$\mu M$]")
    ax3.set_xlim(tspan[0], tspan[1])
    ax3.set_ylim((0.0, 6.0))

# }}}
