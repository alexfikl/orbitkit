# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import replace

import numpy as np

import orbitkit.symbolic.primitives as sym
from orbitkit.codegen.jitcdde import JiTCDDETarget
from orbitkit.models import (
    constant_past_initial_conditions,
    transform_distributed_delay_model,
)
from orbitkit.models.mackey_glass import MackeyGlass1, make_model_from_name
from orbitkit.utils import module_logger, on_ci

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

if not JiTCDDETarget.has_jitcdde():
    log.error("This example requires `jitcdde`.")
    raise SystemExit(0)

# {{{ right-hand side

figname = "Figure2b"
model = make_model_from_name(f"MackeyGlass1977{figname}")
model = replace(model, k=sym.Variable("k"))
assert isinstance(model, MackeyGlass1)

log.info("Model: %s", type(model))
log.info("Equations:\n%s", model)

ext_model = transform_distributed_delay_model(model, 1)
log.info("Model: %s", type(ext_model))
log.info("Equations:\n%s", ext_model)

target = JiTCDDETarget(nlyapunov=1)
code = target.generate_model_code(ext_model, model.n)
integrator = target.compile(code, debug=False)

log.info("\n%s", integrator.f)

# }}}

# {{{ evolve

dt = 0.01
tspan = (0.0, 3000.0)
y0 = constant_past_initial_conditions(ext_model, {"P": np.array([0.1])})

ts = np.arange(tspan[0] + integrator.dde.max_delay, tspan[1], dt)
ys = np.empty(y0.shape + ts.shape, dtype=y0.dtype)

lyap_loc = np.empty((target.nlyapunov, *ts.shape), dtype=y0.dtype)
weights = np.empty(ts.shape, dtype=y0.dtype)
m = int(0.25 * ts.size)

ks = np.linspace(7.0, 25.0, 64)
llyap = np.empty(len(ks))

for i, k in enumerate(ks):
    integrator.set_initial_conditions(y0, tspan[0])
    integrator.set_parameters(k=k)
    integrator.step_on_discontinuities()

    assert integrator.dde.t <= ts[0]
    for n in range(ts.size):
        _, lyap_loc[:, n], weights[n] = integrator.integrate(ts[n])

    # compute largest Lyapunov exponent after transients
    llyap[i] = np.average(lyap_loc[0, -m:], weights=weights[-m:])
    log.info("k %8.5f lyapunov exponent: %+.8e", k, llyap[i])

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
    dirname / f"mackey_glass_{figname.lower()}_lyapunov_exponent",
    figsize=(18, 6),
    overwrite=True,
) as fig:
    ax = fig.gca()

    ax.plot(ks, llyap)
    ax.axhline(0.0, color="k", ls="--")

    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\lambda_{\text{max}}(t)$")

# }}}
