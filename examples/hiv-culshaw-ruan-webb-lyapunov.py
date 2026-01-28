# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.models import transform_distributed_delay_model
from orbitkit.models.hiv import CulshawRuanWebb, make_model_from_name
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

figname = "Figure44"
model = make_model_from_name(f"CulshawRuanWebb2003{figname}")
assert isinstance(model, CulshawRuanWebb)
assert isinstance(model.h, DiracDelayKernel)
assert isinstance(model.h.tau, (int, float))

log.info("Model: %s", type(model))
log.info("Equations:\n%s", model)

ext_model = transform_distributed_delay_model(model, 1)
log.info("Model: %s", type(ext_model))
log.info("Equations:\n%s", ext_model)

from orbitkit.codegen.jitcdde import JiTCDDELyapunovTarget, make_input_variable

target = JiTCDDELyapunovTarget(nlyapunov=4)
source_func = target.lambdify_model(ext_model, model.n)

y = make_input_variable(2)
source = source_func(jitcdde.t, y)
log.info("\n%s", source)

# }}}

# {{{ evolve

tspan = (0.0, 2500.0)
y0 = np.array([5.0e5, 500])

dde = target.compile(source, y, max_delay=model.h.tau)
dde.constant_past(y0, time=tspan[0])
dde.step_on_discontinuities()

dt = 0.01
ts = np.arange(tspan[0] + dde.t, tspan[1], dt)
ys = np.empty(y0.shape + ts.shape, dtype=y0.dtype)

lyap_loc = np.empty((target.nlyapunov, *ts.shape), dtype=y0.dtype)
weights = np.empty(ts.shape, dtype=y0.dtype)

for i in range(ts.size):
    ys[:, i], lyap_loc[:, i], weights[i] = dde.integrate(ts[i])

# compute global Lyapunov exponent
weights = weights.reshape(1, -1)
lyap = np.cumsum(lyap_loc * weights, axis=1) / np.cumsum(weights).reshape(1, -1)
assert lyap.shape == (target.nlyapunov, *ts.shape)

# compute largest Lyapunov exponent after transients
m = int(0.25 * ts.size)
for i in range(target.nlyapunov):
    llyap = np.average(lyap_loc[i, -m:], weights=weights[0, -m:])
    log.info("%d. Lyapunov exponent: %+.8e", i, llyap)

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
    dirname / f"hiv_crw_{figname.lower()}_lyapunov",
    figsize=(18, 6),
    overwrite=True,
) as fig:
    ax = fig.gca()

    ax.plot(ys[0], ys[1])
    ax.plot(y0[0], y0[1], "ko", markersize=10)

    ax.set_xlabel("$C(t)$")
    ax.set_ylabel("$I(t)$")
    ax.set_xlim([-3.0e4, 2.1e6])
    ax.set_ylim([-3.0e4, 1.0e6])

with figure(
    dirname / f"hiv_crw_{figname.lower()}_lyapunov_exponent",
    figsize=(18, 6),
    overwrite=True,
) as fig:
    ax = fig.gca()

    ax.plot(ts[-m:], lyap[0, -m:])
    ax.axhline(0.0, color="k", ls="--")

    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\lambda(t)$")

# }}}
