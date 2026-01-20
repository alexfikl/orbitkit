# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
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
    import jitcode
except ImportError:
    log.error("This example requires `jitcdde` and 'jitcode'.")
    raise SystemExit(0) from None

# FIXME: Figure 5.4 does not match [CulshawRuanWebb2003] very well. Increasing
# k_I' a bit makes it better, but it's not clear why this is necessary.

# {{{ right-hand side

figname = "Figure52"
model = make_model_from_name(f"CulshawRuanWebb2003{figname}")
assert isinstance(model, CulshawRuanWebb)

log.info("Model: %s", type(model))
log.info("Equations:\n%s", model)

ext_model = transform_distributed_delay_model(model, 1)
log.info("Model: %s", type(ext_model))
log.info("Equations:\n%s", ext_model)

# NOTE: Figure 5.2 and Figure 5.4 use the weak Gamma kernel, which is transformed
# into a system of 3 ODEs, so we use `jitcode` instead of `jitcdde` for that.
if figname in {"Figure52", "Figure54"}:
    from orbitkit.codegen.jitcode import JiTCODETarget, make_input_variable

    target = JiTCODETarget()
    source_func = target.lambdify_model(ext_model, model.n)

    y = make_input_variable(2 + 1)
    source = source_func(jitcode.t, y)
else:
    from orbitkit.codegen.jitcdde import JiTCDDETarget, make_input_variable

    target = JiTCDDETarget()
    source_func = target.lambdify_model(ext_model, model.n)

    y = make_input_variable(2)
    source = source_func(jitcdde.t, y)

log.info("\n%s", source)

# }}}

# {{{ evolve

tspan = (0.0, 300.0)

if figname in {"Figure52", "Figure54"}:
    # NOTE: the extra variable in the new system is given by
    #   z = (h * (C * I))(0) = C(0) * I(0)
    # since we assume a constant history for the two variables.
    y0 = np.array([5.0e5, 500, 5.0e5 * 500])

    dde = target.compile(source, y)
    dde.set_initial_value(y0, tspan[0])
else:
    assert isinstance(target, JiTCDDETarget)
    assert isinstance(model.h, DiracDelayKernel)
    assert isinstance(model.h.tau, (int, float))

    y0 = np.array([5.0e5, 500])

    dde = target.compile(source, y, max_delay=model.h.tau)
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

with figure(
    dirname / f"hiv_culshaw_{figname.lower()}",
    figsize=(18, 6),
    overwrite=True,
) as fig:
    ax = fig.gca()

    ax.plot(ys[0], ys[1])
    ax.plot(y0[0], y0[1], "ko", markersize=10)

    for Cstar, Istar in model.fixed_points().values():
        log.info("Fixed point: (%.4e, %.4e)", Cstar, Istar)

        assert isinstance(Cstar, (int, float))
        assert isinstance(Istar, (int, float))
        ax.plot(Cstar, Istar, "ro", markersize=10)

    ax.set_xlabel("$C(t)$")
    ax.set_ylabel("$I(t)$")
    ax.set_xlim([-3.0e4, 2.1e6])
    ax.set_ylim([-3.0e4, 1.0e6])

if figname in {"Figure52", "Figure54"}:
    with figure(
        dirname / f"hiv_culshaw_{figname.lower()}_aux",
        figsize=(18, 6),
        overwrite=True,
    ) as fig:
        ax = fig.gca()

        ax.plot(ts, ys[2])
        ax.set_xlabel("$t$")
        ax.set_ylabel("$z(t)$")

# }}}
