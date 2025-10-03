# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.models.kuramoto import make_model_from_name
from orbitkit.models.symbolic import stringify
from orbitkit.models.targets import NumpyTarget
from orbitkit.utils import module_logger

log = module_logger(__name__)

# FIXME: This seems to reproduce Figure 1e-g more or less perfectly, but does
# not reproduce Figure 1b-d almost at all. It seems like the paper has solutions
# that grow faster and become wavy (e.g. Figure 1c), but we do not get that.

# {{{ create right-hand side

figname = "Figure1c"
model = make_model_from_name(f"Schroder2017{figname}")

log.info("Model: %s", type(model))
log.info("Size:  %d", model.n)

args, exprs = model.symbolify(model.n, full=True)
for i, (name, eq) in enumerate(zip(args[1:], exprs, strict=True)):
    log.info("Eq%d:\n    d%s/dt = %s", i, stringify(name), stringify(eq))

target = NumpyTarget()
source = target.lambdify_model(model, model.n)

# }}}

# {{{ evolve

from scipy.integrate import solve_ivp

tspan = (0.0, 20.0)
tmin_for_plot = 0.0

log.info("tspan: %s", tspan)
log.info("param: %s", figname)
log.info(model)

y0 = np.zeros(model.n)
result = solve_ivp(
    source,
    tspan,
    y0,
    method="RK45",
    atol=1.0e-6,
    rtol=1.0e-8,
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

from orbitkit.visualization import figure, to_color, write_dot_from_adjacency

with figure(
    dirname / f"kuramoto_{figname.lower()}_solution", figsize=(10, 5), overwrite=True
) as fig:
    ax = fig.gca()

    ax.plot(result.t, result.y.T)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\theta_i(t)$")
    if figname in {"Figure1b", "Figure1c", "Figure1d"}:
        ax.set_ylim(-10.0, 10.0)
    else:
        ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(tspan)

write_dot_from_adjacency(
    dirname / f"kuramoto_{figname.lower()}.dot",
    model.K,
    nodenames=tuple(f"{omega:g}" for omega in model.omega),
    nodecolors=to_color(model.omega),
    overwrite=True,
)

# }}}
