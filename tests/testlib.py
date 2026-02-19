# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace

import numpy as np

from orbitkit.models import Model


def get_model_from_module(module_name: str, model_name: str, n: int) -> Model:
    # construct a dummy all-to-all connectivity matrix for the models that need it
    A = np.ones((n, n)) - np.eye(n)
    model: Model

    if module_name == "fitzhugh_nagumo":
        from orbitkit.models import fitzhugh_nagumo

        model = replace(fitzhugh_nagumo.make_model_from_name(model_name), G=A)
    elif module_name == "hiv":
        from orbitkit.models import hiv

        model = hiv.make_model_from_name(model_name)
    elif module_name == "kuramoto":
        from orbitkit.models import kuramoto

        model = kuramoto.make_model_from_name(model_name)
    elif module_name == "wang_rinzel":
        from orbitkit.models import wang_rinzel

        model = replace(wang_rinzel.make_model_from_name(model_name), A=A)
    elif module_name == "wang_buzsaki":
        from orbitkit.models import wang_buzsaki

        model = replace(wang_buzsaki.make_model_from_name(model_name), A=A)
    elif module_name == "pfeuty":
        from orbitkit.models import pfeuty

        model = replace(pfeuty.make_model_from_name(model_name), A_inh=A, A_gap=A)
    elif module_name == "wilson_cowan":
        from orbitkit.models import wilson_cowan

        model = wilson_cowan.make_model_from_name(model_name)
        model = replace(
            model,
            E=replace(
                model.E,
                weights=(*model.E.weights[:-1], (A, 0)),
                forcing=np.full(n, model.E.forcing[0]),
            ),
            I=replace(model.I, forcing=np.full(n, model.I.forcing[0])),
        )
    else:
        raise ValueError(f"unknown module name: '{module_name}'")

    return model
