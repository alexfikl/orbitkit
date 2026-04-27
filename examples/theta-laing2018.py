# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.codegen.jitcode import JiTCODETarget
from orbitkit.models.theta import find_fixed_points, make_model_from_name
from orbitkit.utils import module_logger, on_ci, tictoc

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

if not JiTCODETarget.has_jitcode():
    log.error("This example requires `jitcode`.")
    raise SystemExit(0)

# {{{ create right-hand side

figname = "Figure2b"
model = make_model_from_name(f"Laing2018{figname}")

log.info("Param: %s", figname)
log.info("Model: %s", type(model))
log.info("Equations:\n%s", model)

with tictoc("codegen"):
    target = JiTCODETarget()
    code = target.generate_model_code(model, (model.n, model.n))

# }}}

# {{{ evolve

from orbitkit.utils import generate_random_points_in_disk

dt = 0.025
tspan = (0.0, 1000.0)

nruns = 100 if figname == "Figure2a" else 50
y0 = generate_random_points_in_disk(nruns, (0.1, 0.9), dtype=np.float64, rng=rng)

with tictoc("compile"):
    integrator = target.compile(code, method="RK45", debug=False)

with tictoc("evolve"):
    t = np.arange(tspan[0], tspan[1], dt)
    y = np.empty((nruns, t.size, 2), dtype=y0.dtype)

    for i in range(nruns):
        integrator.set_initial_conditions(y0[:, i], tspan[0])

        for j in range(t.size):
            y[i, j], _, _ = integrator.integrate(t[j])

# }}}

# {{{ plot

if on_ci():
    raise SystemExit(0)

try:
    import matplotlib.pyplot as mp  # noqa: F401
except ImportError:
    raise SystemExit(0) from None

from orbitkit.visualization import figure, get_color_cycle, set_plotting_defaults

dirname = pathlib.Path(__file__).parent
set_plotting_defaults()

with figure(dirname / f"theta_laing2018_{figname.lower()}", overwrite=True) as fig:
    ax = fig.gca()
    bdry_color, fp1_color, fp2_color, *_ = get_color_cycle()

    # plot trajectories
    for i in range(nruns):
        ax.plot(y[i, :, 0], y[i, :, 1], color="k", lw=1)

    # plot the unit circle
    from matplotlib.patches import Circle

    circle = Circle((0.0, 0.0), 1.0, color=bdry_color, lw=2, fill=False, zorder=10)
    ax.add_patch(circle)

    # plot fixed points
    fp = find_fixed_points(model)
    ax.plot(fp.on_circle.real, fp.on_circle.imag, "s", color=fp1_color, zorder=11)
    ax.plot(fp.in_disk.real, fp.in_disk.imag, "^", color=fp2_color, zorder=11)

    ax.set_aspect("equal")

# }}}
