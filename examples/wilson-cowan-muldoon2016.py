# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import pathlib
import zipfile
from dataclasses import replace

import numpy as np

from orbitkit.models import transform_distributed_delay_model
from orbitkit.models.wilson_cowan import make_model_from_name
from orbitkit.symbolic.primitives import DiracDelayKernel
from orbitkit.utils import download_from_data_dryad, load_from_mat, module_logger, on_ci

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)

if on_ci():
    raise SystemExit(0)

try:
    import jitcdde
except ImportError:
    log.error("This example requires 'jitcdde'.")
    raise SystemExit(0) from None

# {{{ download datasets

# https://datadryad.org/dataset/doi:10.5061/dryad.8g4vp
filename = pathlib.Path(__file__).parent / "muldoon2016.zip"
dirname = filename.parent / filename.stem

if not filename.exists():
    token = os.environ.get("DATADRYAD_TOKEN")
    if token is None:
        log.error(
            "This example requires the dataset from "
            "'https://datadryad.org/dataset/doi:10.5061/dryad.8g4vp'. Please "
            "acquire it from there and put it in 'examples/muldoon2016.zip'. "
            "Alternatively, obtain an API token from datadryad.org and export it "
            "as 'export DATADRYAD_TOKEN=<token>'."
        )
        raise SystemExit(1)

    download_from_data_dryad(filename, "10.5061/dryad.8g4vp", token)

if not dirname.exists():
    dirname.mkdir(exist_ok=True)

    with zipfile.ZipFile(filename, "r") as z:
        z.extractall(dirname)

    with zipfile.ZipFile(dirname / "connectivity_matrices.zip") as z:
        z.extractall(dirname)

log.info("Using datasets from '%s'.", dirname)

dataset = load_from_mat(dirname / "subject1_scan1.mat")
A = dataset.connectivity_density_matrix
# FIXME: we do not support array delays yet
tau = round(np.mean(dataset.delay_matrix), ndigits=5)

assert A.shape[0] == A.shape[1]
n = A.shape[0]

from orbitkit.adjacency import stringify_adjacency

print(stringify_adjacency(A, fmt="tight"))

# }}}

# {{{ create right-hand side

figname = ""
model = make_model_from_name(f"MuldoonPasqualetti2016{figname}")
model = replace(
    model,
    E=replace(
        model.E,
        kernels=(*model.E.kernels[:-1], DiracDelayKernel(tau)),
        weights=(*model.E.weights[:-1], (A, 0)),
        forcing=np.full(n, model.E.forcing[0]),
    ),
    I=replace(model.I, forcing=np.full(n, model.I.forcing[0])),
)
ext_model = transform_distributed_delay_model(model, model.n)

log.info("Model: %s", type(model))
log.info("Size:  %d", model.n)
log.info("Equations:\n%s", model)

# }}}

# {{{ codegen

from orbitkit.codegen.jitcdde import JiTCDDETarget, make_input_variable

target = JiTCDDETarget()
source_func = target.lambdify_model(ext_model, model.n)

y = make_input_variable(2 * model.n)
source = source_func(jitcdde.t, y)

dde = target.compile(source, y, max_delay=tau)

# }}}

# {{{ evolve

tspan = (0.0, 1000.0)

y0 = np.concatenate([
    0.1 + 0.1 * rng.random(model.n),
    0.0 + 0.1 * rng.random(model.n),
])
dde.constant_past(y0, time=tspan[0])

# NOTE: using adjust_diff seems to give results a lot closer to [ContiGorder2019].
# Maybe that's what MATLAB uses as well? Or similar at least..
# dde.step_on_discontinuities()
dde.adjust_diff()

dt = (tspan[1] - tspan[0]) / 100000
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

with figure(
    dirname / f"wilson_cowan_muldoon2016_{dataset.filename.stem}",
    figsize=(10, 5),
    overwrite=True,
) as fig:
    ax = fig.gca()

    ax.plot(ts, ys[: model.n].T)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$E(t)$")
    ax.set_xlim(tspan)

# }}}
