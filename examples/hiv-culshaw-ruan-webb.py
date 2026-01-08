# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

# from orbitkit.codegen.numpy import NumpyTarget
from orbitkit.models import transform_distributed_delay_model
from orbitkit.models.hiv import CulshawRuanWebb, make_model_from_name
from orbitkit.utils import module_logger

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)


# {{{ right-hand side

figname = "Figure32"
model = make_model_from_name(f"CulshawRuanWebb2003{figname}")
assert isinstance(model, CulshawRuanWebb)

log.info("Model: %s", type(model))
log.info("Equations:\n%s", model)

ext_model = transform_distributed_delay_model(model, 1)
log.info("Model: %s", type(ext_model))
log.info("Equations:\n%s", ext_model)

# target = NumpyTarget()
# source = target.lambdify_model(model, model.n)

# }}}
