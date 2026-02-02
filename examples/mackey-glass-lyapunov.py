# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import replace

import numpy as np

import orbitkit.symbolic.primitives as sym
from orbitkit.models import transform_distributed_delay_model
from orbitkit.models.mackey_glass import MackeyGlass1, make_model_from_name
from orbitkit.symbolic.primitives import DiracDelayKernel
from orbitkit.utils import module_logger, on_ci

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

try:
    import jitcdde
except ImportError:
    log.error("This example requires `jitcdde`.")
    raise SystemExit(0) from None


# {{{ right-hand side

figname = "Figure2b"
model = make_model_from_name(f"MackeyGlass1977{figname}")
model = replace(model, k=sym.Variable("k"))

assert isinstance(model, MackeyGlass1)
assert isinstance(model.h, DiracDelayKernel)
assert isinstance(model.h.tau, (int, float))

log.info("Model: %s", type(model))
log.info("Equations:\n%s", model)

ext_model = transform_distributed_delay_model(model, 1)
log.info("Model: %s", type(ext_model))
log.info("Equations:\n%s", ext_model)

from orbitkit.codegen.jitcdde import JiTCDDELyapunovTarget, make_input_variable

target = JiTCDDELyapunovTarget(nlyapunov=1)
source_func = target.lambdify_model(ext_model, model.n)

y = make_input_variable(1)
source = source_func(jitcdde.t, y)
log.info("\n%s", source)

tau = model.h.tau
dde = target.compile(source, y, max_delay=tau, parameters=("k",))

# }}}

# {{{ evolve

dt = 0.01
tspan = (0.0, 3000.0)

y0 = np.array([0.1])
ts = np.arange(tspan[0] + tau, tspan[1], dt)
ys = np.empty(y0.shape + ts.shape, dtype=y0.dtype)

lyap_loc = np.empty((target.nlyapunov, *ts.shape), dtype=y0.dtype)
weights = np.empty(ts.shape, dtype=y0.dtype)
m = int(0.25 * ts.size)

ks = np.linspace(7.0, 25.0, 64)
llyap = np.empty(len(ks))

for i, k in enumerate(ks):
    dde.constant_past(y0, time=tspan[0])
    dde.set_parameters((k,))
    dde.step_on_discontinuities()

    assert dde.t <= ts[0]
    for n in range(ts.size):
        _, lyap_loc[:, n], weights[n] = dde.integrate(ts[n])

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
