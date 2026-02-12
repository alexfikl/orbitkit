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

figname = "Figure3b"
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

if figname.startswith("Figure2"):
    tspan = (0.0, 100.0)
elif figname.startswith("Figure3") or figname.startswith("Figure4"):
    tspan = (0.0, 140.0)
else:
    raise ValueError(f"unsupported figure: {figname!r}")

y0 = np.array([0.25] * model.n + [0.75] * model.n)

# NOTE: using adjust_diff seems to give results a lot closer to [ContiGorder2019].
# Maybe that's what MATLAB uses as well? Or similar at least..
dde.constant_past(y0, time=tspan[0])
# dde.step_on_discontinuities()
dde.adjust_diff()

dt = 0.001
ts = np.arange(tspan[0], tspan[1], dt)
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

    (line,) = ax.plot(ts, ys[0], label=r"$\boldsymbol{E}(t)$")
    for i in range(1, model.n):
        ax.plot(ts, ys[i], color=line.get_color())

    (line,) = ax.plot(ts, ys[model.n], ls="--", label=r"$\boldsymbol{I}(t)$")
    for i in range(model.n + 1, 2 * model.n):
        ax.plot(ts, ys[i], ls="--", color=line.get_color())

    ax.set_xlabel("$t$")
    ax.set_xlim(tspan)
    ax.set_ylim([0.0, 1.0])
    ax.legend()

# }}}
