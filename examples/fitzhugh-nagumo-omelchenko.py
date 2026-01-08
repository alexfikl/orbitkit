# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from orbitkit.codegen.jax import JaxTarget
from orbitkit.models.fitzhugh_nagumo import (
    FitzHughNagumoOmelchenko,
    make_model_from_name,
)
from orbitkit.utils import module_logger

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    log.error("This example requires `jax` and `diffrax`.")
    raise SystemExit(0) from None

# FIXME: this will not look like in the paper because we do not incorporate the
# delays. We could do that in the future, since diffrax wants to support it.

# {{{ create right-hand side

figname = "Figure1"
model = make_model_from_name(f"Omelchenko2019{figname}")
assert isinstance(model, FitzHughNagumoOmelchenko)

log.info("Model: %s", type(model))
log.info("Size:  %d", model.n)
log.info("Equations:\n%s", model)

target = JaxTarget()
source = target.lambdify_model(model, model.n)

# }}}


# {{{ evolve

try:
    from diffrax import (  # ty: ignore[unresolved-import]
        Dopri5,
        ODETerm,
        PIDController,
        SaveAt,
        diffeqsolve,
    )
except ImportError:
    log.error("This example requires `jax` and `diffrax`.")
    raise SystemExit(0) from None

tspan = (0.0, 120.0)
tmin_for_plot = 112.0

log.info("tspan: %s", tspan)
log.info("param: %s", figname)
log.info(model)

# NOTE: Figure1 just says that the initial conditions are "random". We use the
# [-2.0, 2.0] interval to match the magnitude of he solutions at T
y0 = jax.device_put(
    np.hstack([
        rng.uniform(-2.0, 2.0, size=model.n),
        rng.uniform(-2.0, 2.0, size=model.n),
    ])
)

result = diffeqsolve(
    ODETerm(lambda t, y, args: source(t, y)),  # type: ignore[arg-type,unused-ignore]
    Dopri5(),
    t0=tspan[0],
    t1=tspan[1],
    dt0=0.01,
    y0=y0,
    max_steps=2 * 4096,
    saveat=SaveAt(ts=jnp.linspace(*tspan, 12000)),
    stepsize_controller=PIDController(atol=1.0e-5, rtol=1.0e-5),
)

ts = jax.device_get(result.ts)
ys = jax.device_get(result.ys.T)

# }}}


# {{{ plot

try:
    import matplotlib.pyplot as mp  # noqa: F401
except ImportError:
    raise SystemExit(0) from None

from orbitkit.visualization import figure, set_plotting_defaults

dirname = pathlib.Path(__file__).parent
set_plotting_defaults()

mask = ts > tmin_for_plot
ns = np.arange(model.n)

with figure(
    dirname / f"fitzhugh_nagumo_omelchenko_{figname.lower()}_solution",
    figsize=(10, 3),
    overwrite=True,
) as fig:
    ax = fig.gca()

    t, n = np.meshgrid(ts[mask], ns)
    im = ax.contourf(n, t, ys[: model.n, mask], cmap="jet")

    ax.set_xlabel("$i$")
    ax.set_ylabel("$t$")
    # ax.set_xlim(0, model.n)
    # ax.set_ylim(tmin_for_plot, tspan[-1])
    fig.colorbar(im, ax=ax)

with figure(
    dirname / f"fitzhugh_nagumo_omelchenko_{figname.lower()}_final",
    figsize=(10, 3),
    overwrite=True,
) as fig:
    ax = fig.gca()

    ax.plot(ns, ys[: model.n, -1], "o", markersize=5)
    ax.set_xlabel("$i$")
    ax.set_ylabel("$u_i$")

# with figure(
#     dirname / f"fitzhugh_nagumo_omelchenko_{figname.lower}_phase_velocity",
#     overwrite=True,
# ) as fig:
#     ax = fig.gca()

# }}}
