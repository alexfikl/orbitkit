# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from orbitkit.typing import Array
from orbitkit.utils import enable_test_plotting, module_logger
from orbitkit.visualization import figure, set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)
set_plotting_defaults()


# {{{ test_periodic_behavior_culshaw


def visualize_behavior(
    ts: Array,
    ys: Array,
    basename: str,
    *,
    lcmethod: str,
    ylabels: tuple[str, ...] | None = None,
) -> None:
    from orbitkit.cycles import (
        detect_cycle_auto_correlation,
        detect_cycle_harmonic,
    )

    if ylabels is None:
        ylabels = tuple(f"$x_{{{i}}}$" for i in range(ys.shape[0]))

    n = int(0.25 * ts.size)
    ys = ys[:, -n:]
    ts = ts[-n:]
    dt = np.min(np.diff(ts))

    with figure(TEST_DIRECTORY / f"{basename}_solution", normalize=True) as fig:
        ax = fig.gca()

        for i in range(ys.shape[0]):
            ax.plot(ts, ys[i], label=ylabels[i])
        ax.set_xlabel("$t$")
        ax.legend()

    with figure(TEST_DIRECTORY / f"{basename}_metric", normalize=True) as fig:
        ax = fig.gca()

        if lcmethod == "acf":
            for i in range(ys.shape[0]):
                result = detect_cycle_auto_correlation(ys[i])
                log.info("Peaks: %s", result.peaks)

                ax.plot(result.corr, label=ylabels[i])
                ax.plot(result.peaks, result.corr[result.peaks], "ro")

            ax.set_xlabel("$n$")
            ax.set_ylim((-1.0, 1.0))
            ax.legend()
        elif lcmethod == "harm":
            ratio = np.zeros(ys.shape[0])
            for i in range(ys.shape[0]):
                result = detect_cycle_harmonic(ys[i], fs=1.0 / dt)
                ratio[i] = abs(1 - result.harmonic_energy / result.total_energy)
                log.info(
                    "PSD: harmonic %.8e fraction %.8e", result.harmonic_energy, ratio[i]
                )

            ax.semilogy(ratio, "o-")
            ax.set_ylim((0.0, 1.0))
            ax.set_xlabel("$i$")
            ax.set_ylabel("$r$")
        else:
            raise AssertionError(lcmethod)

    if lcmethod == "harm":
        with figure(TEST_DIRECTORY / f"{basename}_psd", normalize=True) as fig:
            ax = fig.gca()

            for i in range(ys.shape[0]):
                result = detect_cycle_harmonic(ys[i], fs=1.0 / dt)
                ax.plot(result.freq, result.psd[i], label=ylabels[i])

            ax.set_xlabel("$k$")
            ax.legend()


@pytest.mark.parametrize("figname", ["Figure42", "Figure44"])
def test_periodic_behavior_culshaw(figname: str) -> None:
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

    visualize_behavior(
        ts,
        ys,
        f"test_behavior_culshaw_{figname}",
        lcmethod=lcmethod,
        ylabels=("$C(t)$", "$I(t)$"),
    )


# }}}


# {{{ test_periodic_behavior_van_der_pol


@pytest.mark.parametrize("mu", [0.1, 5.0])
def test_periodic_behavior_van_der_pol(mu: float) -> None:
    pytest.importorskip("pymbolic")
    rng = np.random.default_rng(seed=32)

    from orbitkit.models.van_der_pol import VanDerPol

    model = VanDerPol(mu=mu, amplitude=1.0, omega=1.0)
    log.info("Model: %s", type(model))
    log.info("Equations:\n%s", model)

    from orbitkit.codegen.numpy import NumpyTarget

    target = NumpyTarget()
    source = target.lambdify_model(model, model.n)

    from scipy.integrate import solve_ivp

    tspan = (0.0, 100.0)
    y0 = rng.random(size=2)

    result = solve_ivp(
        source,
        tspan,
        y0,
        method="RK45",
        atol=1.0e-8,
        rtol=1.0e-10,
        max_step=0.1,
    )

    from orbitkit.behavior import Behavior, determine_behavior

    lcmethod = "harm"
    b = determine_behavior(result.y, lcmethod=lcmethod)
    log.info("Behavior: %s", b)

    assert b == Behavior.Periodic

    if not enable_test_plotting():
        return

    visualize_behavior(
        result.t,
        result.y,
        f"test_behavior_van_der_pol_{mu}",
        lcmethod=lcmethod,
        ylabels=("$x$", "$y$"),
    )


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
