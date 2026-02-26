# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from orbitkit.utils import enable_test_plotting, module_logger
from orbitkit.visualization import figure, set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)
set_plotting_defaults()


# {{{ test_periodic_behavior


@pytest.mark.parametrize("figname", ["Figure42", "Figure44"])
def test_periodic_behavior(figname: str) -> None:
    # {{{ simulate example

    pytest.importorskip("pymbolic")
    jitcdde = pytest.importorskip("jitcdde")

    from orbitkit.models import transform_distributed_delay_model
    from orbitkit.models.hiv import CulshawRuanWebb, make_model_from_name

    model = make_model_from_name(f"CulshawRuanWebb2003{figname}")
    ext_model = transform_distributed_delay_model(model, 1)
    assert isinstance(model, CulshawRuanWebb)

    log.info("Model: %s", type(ext_model))
    log.info("Equations:\n%s", ext_model)

    from orbitkit.codegen.jitcdde import JiTCDDETarget, make_input_variable

    target = JiTCDDETarget()
    source_func = target.lambdify_model(ext_model, 1)

    y = make_input_variable(2)
    source = source_func(jitcdde.t, y)

    tspan = (0.0, 600.0)
    y0 = np.array([5.0e5, 500])

    dde = target.compile(source, y, max_delay=model.h.avg)  # ty: ignore[invalid-argument-type]
    dde.constant_past(y0, tspan[0])
    dde.adjust_diff()

    dt = 0.01
    ts = np.arange(tspan[0], tspan[1], dt)
    ys = np.empty(y0.shape + ts.shape, dtype=y0.dtype)

    for i in range(ts.size):
        ys[:, i] = dde.integrate(ts[i])

    # }}}

    # {{{ check periodicity

    from orbitkit.behavior import Behavior, determine_behavior

    lcmethod = "harm"
    b = determine_behavior(ys, lcmethod=lcmethod)
    log.info("Behavior: %s", b)

    if figname == "Figure42":
        assert b == Behavior.FixedPoint
    elif figname == "Figure44":
        assert b == Behavior.Periodic
    else:
        raise ValueError(f"unsupported parameters: {figname}")

    # }}}

    if not enable_test_plotting():
        return

    from orbitkit.cycles import (
        detect_cycle_auto_correlation,
        detect_cycle_harmonic,
    )

    n = int(0.25 * ts.size)
    ys = ys[:, -n:]
    ts = ts[-n:]

    with figure(
        TEST_DIRECTORY / f"test_behavior_culshaw_{figname}", normalize=True
    ) as fig:
        ax = fig.gca()

        ax.plot(ts, ys[0], label="$C(t)$")
        ax.plot(ts, ys[1], label="$I(t)$")
        ax.set_xlabel("$t$")
        ax.legend()

    with figure(
        TEST_DIRECTORY / f"test_behavior_culshaw_{figname}_periodic", normalize=True
    ) as fig:
        ax = fig.gca()
        labels = ["$C(t)$", "$I(t)$"]

        if lcmethod == "acf":
            for i in range(ys.shape[0]):
                result = detect_cycle_auto_correlation(ys[i])
                log.info("Peaks: %s", result.peaks)

                ax.plot(result.corr, label=labels[i])
                ax.plot(result.peaks, result.corr[result.peaks], "ro")

            ax.set_xlabel("$n$")
            ax.set_ylim((-1.0, 1.0))
        elif lcmethod == "harm":
            ratio = np.zeros(ys.shape[0])
            for i in range(ys.shape[0]):
                result = detect_cycle_harmonic(ys[i], fs=1.0 / dt)
                ratio[i] = 1 - result.harmonic_energy / result.total_energy
                log.info(
                    "PSD: harmonic %.8e fraction %.8e", result.harmonic_energy, ratio[i]
                )

            ax.semilogy(ratio, "o-")
            ax.set_ylim((0.0, 1.0))
            ax.set_xlabel("$i$")
            ax.set_ylabel("$r$")
            ax.legend()
        else:
            raise AssertionError(lcmethod)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
