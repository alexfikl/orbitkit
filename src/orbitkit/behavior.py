# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import enum
from typing import Any, Literal

import numpy as np

from orbitkit.typing import Array1D, Array2D
from orbitkit.utils import module_logger

log = module_logger(__name__)


@enum.unique
class Behavior(enum.Enum):
    """Macro view of the behavior of a dynamical system."""

    Unknown = enum.auto()
    """System has an unknown behavior that could not be determined."""

    Divergent = enum.auto()
    """System has diverged to infinity."""
    Chaotic = enum.auto()
    """System is chaotic, i.e. it has a positive Lyapunov exponent."""
    FixedPoint = enum.auto()
    """System has reached a fixed point (or steady state)."""
    Periodic = enum.auto()
    """System has reached a cycle, i.e. all components are periodic, but not
    necessarily synchronized.
    """


def is_divergent(x: Array1D[np.floating[Any]]) -> bool:
    """Check if the system is divergent."""
    return not np.isfinite(x)


def is_fixed_point(
    x: Array1D[np.floating[Any]],
    *,
    xtol: float = 1.0e-3,
    gtol: float = 1.0e-4,
    atol: float = 1.0e-12,
) -> bool:
    """Check if the given time series is converged to a fixed point.

    We check if the time series has reached a fixed point by looking at two
    slightly different metrics:
    1. The standard deviation of the time series. This should be very small for
       any constant time series.
    2. The slope of a least squares linear fit. This should catch the case where
       the solution is actually just slowly drifting (e.g. :math:`y = 0.01 x`).

    :arg x: a one-dimensional time series. If the time series is naturally
        multi-dimensional, a norm in the dimension should be performed before.
    :arg xtol: a relative tolerance in the standard deviation of the time series.
    :arg gtol: a relative tolerance in the slope of the time series.
    :arg atol: an absolute tolerance used when the fixed point is close to zero,
        making the relative tolerances unreliable.
    """
    # normalization constant: norm at the last time step
    mu = np.linalg.norm(x, ord=np.inf)
    mu = mu if mu > 1.0e-8 else 1.0

    # normalize tolerances
    xtol *= mu
    gtol *= mu / x.shape[0]

    # 1. Compute the standard deviation of the time series
    std = np.std(x)

    # 2. Look at the global slope trend
    n = np.arange(x.size)
    slope, _ = np.polyfit(n, x, 1)
    slope = abs(slope)

    log.info(
        "Fixed point: std %.8e (xtol %.8e) slope %.8e (gtol %.8e).",
        std,
        xtol,
        slope,
        gtol,
    )

    return (std < atol or std < xtol) and (slope < atol or slope < gtol)


def is_periodic(
    x: Array1D[np.floating[Any]],
    *,
    rtol: float = 5.0e-1,
    method: Literal["acf", "harm"] = "harm",
) -> bool:
    """Check if the given time series is periodic (has reached a limit cycle)."""
    from orbitkit.cycles import (
        is_limit_cycle_auto_correlation,
        is_limit_cycle_harmonic,
    )

    if method == "acf":
        return is_limit_cycle_auto_correlation(x, eps=rtol)
    elif method == "harm":
        return is_limit_cycle_harmonic(x, eps=rtol)
    else:
        raise ValueError(f"unknown periodicity checking method: {method!r}")


def determine_behavior(
    x: Array1D[np.floating[Any]] | Array2D[np.floating[Any]],
    *,
    nwindow: int | None = None,
    fptol: float = 1.0e-3,
    lctol: float | None = None,
    lcmethod: Literal["acf", "harm"] = "harm",
) -> Behavior:
    """Determine the coarse behavior of the time series *x*.

    For periodicity checking, we use the following methods:
    * ``acf``: :func:`~orbitkit.cycles.detect_cycle_auto_correlation`.
    * ``harm``: :func:`~orbitkit.cycles.detect_cycle_harmonic`.

    Note that the *lctol* tolerance has different meanings for all of these methods.
    Therefore, it should be chosen carefully and there is no reasonable default value.

    :arg x: an array of shape ``([d, ]n)``, where :math:`d` is the dimension of the
        state space and :math:`n` is the time dimension. One-dimensional systems
        can forgo the first dimension.
    :arg nwindow: if given, we only check the last *nwindow* values. By default,
        this looks at the last quarter, i.e. :math:`0.25 n` of the time series.
    :arg fptol: (relative) tolerance used to determine if the time series has
        converged to a fixed point (i.e. reached a steady state).
    :arg lctol: (relative) tolerance used to determine if the time series has
        reached a limit cycle.
    :arg lcmethod: method used to check the periodicity of each component.
    """

    if x.ndim == 1:
        x = x.reshape(1, -1)

    if nwindow is None:
        nwindow = int(0.25 * x.shape[1])
    xd = x[:, -nwindow:]

    if lctol is None:
        lctol = {
            "acf": 5.0e-1,
            "harm": 1.0e-3,
        }.get(lcmethod, 1.0e-3)

    # NOTE: do not be tempted to look at something like norm(x, axis=0) to
    # determine the behavior. This can fail in many common cases, e.g. a signal
    # like `[sin(t), cos(t)]` will be determined as a "fixed point" incorrectly.

    bs = set()
    for x_i in xd:
        if not np.all(np.isfinite(x_i)):
            return Behavior.Divergent

        if is_fixed_point(x_i, xtol=fptol, gtol=fptol):
            bs.add(Behavior.FixedPoint)
        elif is_periodic(x_i, rtol=lctol, method=lcmethod):
            bs.add(Behavior.Periodic)
        else:
            bs.add(Behavior.Unknown)

    if Behavior.Unknown in bs:
        return Behavior.Unknown
    elif Behavior.Divergent in bs:
        # NOTE: this won't happen, but for completeness..
        return Behavior.Divergent
    elif bs == {Behavior.FixedPoint}:
        return Behavior.FixedPoint
    elif not (bs - {Behavior.Periodic, Behavior.FixedPoint}):
        return Behavior.Periodic
    else:
        raise ValueError(f"unsupported combination of behaviors: {bs}")
