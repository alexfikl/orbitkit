# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.codegen.jitcdde import JiTCDDETarget
from orbitkit.models import (
    constant_past_initial_conditions,
    transform_distributed_delay_model,
)
from orbitkit.models.wilson_cowan import make_model_from_name
from orbitkit.utils import module_logger, on_ci

# NOTE:
# - For Figure3, seed=42 gives (a) and seed=43 gives (b)

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

if not JiTCDDETarget.has_jitcdde():
    log.error("This example requires 'jitcdde'.")
    raise SystemExit(0)

# {{{ create right-hand side

figname = "Figure9a"
model = make_model_from_name(f"CoombesLaing2009{figname}")

log.info("Model: %s", type(model))
log.info("Size:  %d", model.n)
log.info("Equations:\n%s", model)

ext_model = transform_distributed_delay_model(model, model.n)
log.info("Model: %s", type(ext_model))
log.info("Equations:\n%s", ext_model)

# }}}

# {{{ codegen

target = JiTCDDETarget()
code = target.generate_model_code(ext_model, model.n)
integrator = target.compile(code, debug=False)

log.info("\n%s", integrator.f)

# }}}

# {{{ evolve

if figname.startswith("Figure9"):  # noqa: SIM108
    tspan = (0.0, 45.0)
else:
    tspan = (0.0, 15.0)

y0 = constant_past_initial_conditions(
    ext_model,
    {
        "E": 0.25 + 0.1 * rng.random(model.n),
        "I": 0.75 + 0.1 * rng.random(model.n),
    },
)
integrator.set_initial_conditions(y0, t=tspan[0])

# NOTE: using adjust_diff seems to give results a lot closer to [ContiGorder2019].
# Maybe that's what MATLAB uses as well? Or similar at least..
# integrator.step_on_discontinuities()
integrator.adjust_diff()

dt = (tspan[1] - tspan[0]) / 10000
ts = np.arange(tspan[0], tspan[1], dt)
ys = np.empty(y0.shape + ts.shape, dtype=y0.dtype)

for i in range(ts.size):
    ys[:, i], _, _ = integrator.integrate(ts[i])

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

if figname.startswith("Figure9"):
    with figure(
        dirname / f"wilson_cowan_coombes2009_{figname.lower()}", overwrite=True
    ) as fig:
        ax = fig.gca()

        mask = ts > 10.0
        ax.plot(ys[0, mask], ys[1, mask])

        ax.set_xlabel("$E(t)$")
        ax.set_ylabel("$I(t)$")
else:
    with figure(
        dirname / f"wilson_cowan_coombes2009_{figname.lower()}",
        figsize=(10, 5),
        overwrite=True,
    ) as fig:
        ax = fig.gca()

        (line,) = ax.plot(ts, ys[0], label=r"$\boldsymbol{E}(t)$")
        for i in range(1, model.n):
            ax.plot(ts, ys[i], ls="--", color=line.get_color())

        (line,) = ax.plot(ts, ys[model.n], label=r"$\boldsymbol{I}(t)$")
        for i in range(model.n + 1, 2 * model.n):
            ax.plot(ts, ys[i], ls="--", color=line.get_color())

        ax.set_xlabel("$t$")
        ax.set_xlim(tspan)
        ax.set_ylim([0.0, 1.0])
        ax.legend()

# }}}
