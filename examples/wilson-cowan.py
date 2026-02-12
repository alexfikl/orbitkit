# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.models import transform_distributed_delay_model
from orbitkit.models.wilson_cowan import make_model_from_name
from orbitkit.utils import module_logger, on_ci

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

try:
    import jitcdde
except ImportError:
    log.error("This example requires 'jitcdde'.")
    raise SystemExit(0) from None

# {{{ create right-hand side

figname = "Figure2b"
model = make_model_from_name(f"ContiGorder2019{figname}")

log.info("Model: %s", type(model))
log.info("Size:  %d", model.n)
log.info("Equations:\n%s", model)

ext_model = transform_distributed_delay_model(model, model.n)
log.info("Model: %s", type(ext_model))
log.info("Equations:\n%s", ext_model)

# }}}

# {{{ codegen

from orbitkit.codegen.jitcdde import JiTCDDETarget, make_input_variable

target = JiTCDDETarget()
source_func = target.lambdify_model(ext_model, model.n)

y = make_input_variable(2 * model.n)
source = source_func(jitcdde.t, y)

log.info("\n%s", source)

max_delay = max(  # ty: ignore[no-matching-overload]
    *(h.avg for h in model.E.kernels),
    *(h.avg for h in model.I.kernels),
)
dde = target.compile(source, y, max_delay=max_delay)

# }}}

# {{{ evolve

tspan = (0.0, 100.0)
y0 = np.array([0.25, 0.25, 0.75, 0.75])
dde.constant_past(y0, time=tspan[0])
dde.step_on_discontinuities()

dt = 0.001
ts = np.arange(tspan[0] + dde.t, tspan[1], dt)
ys = np.empty(y0.shape + ts.shape, dtype=y0.dtype)

for i in range(ts.size):
    ys[:, i] = dde.integrate(ts[i])

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

with figure(dirname / f"wilson_cowan_{figname.lower()}", overwrite=True) as fig:
    ax = fig.gca()

    (line,) = ax.plot(ts, ys[0], label="$E_1(t)$")
    ax.plot(ts, ys[1], ls="--", color=line.get_color(), label="$E_2(t)$")
    (line,) = ax.plot(ts, ys[2], label="$I_1(t)$")
    ax.plot(ts, ys[3], ls="--", color=line.get_color(), label="$I_2(t)$")

    ax.set_xlabel("$t$")
    ax.set_xlim(tspan)
    ax.set_ylim([0.0, 1.0])
    ax.legend()

# }}}
