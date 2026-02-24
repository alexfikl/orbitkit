# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import enum

import numpy as np

from orbitkit.typing import Array
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
    necessarily synchornized.
    """


def is_divergent(x: Array) -> bool:
    """Check if the system is divergent."""
    return not np.isfinite(x)


def is_fixed_point(
    x: Array,
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
    mu = np.linalg.norm(x[-1], ord=np.inf)
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
        "Fixed point std %.8e (xtol %.8e) slope %.8e (gtol %.8e).",
        std,
        xtol,
        slope,
        gtol,
    )

    return (std < atol or std < xtol) and (slope < atol or slope < gtol)


def is_periodic(
    x,
    *,
    rtol: float = 1.0e-3,
    prominence: float | None = None,
    distance: float | None = None,
) -> bool:
    """Check if the given time series is periodic (has reached a limit cycle)."""
    from scipy.signal import correlate, find_peaks

    # 1. compute the auto-correlation
    corr = correlate(x, x, mode="full")[x.size - 1 :]
    corr /= corr[0]

    # 2. find peaks in the auto-correlation
    if distance is None:
        below_zero, _ = np.where(corr < 0)
        first_min = below_zero[0] if below_zero.size > 0 else 5
        distance = max(1.0, int(0.5 * first_min))

    if prominence is None:
        prominence = 0.5 * np.std(corr)

    peaks, _ = find_peaks(corr, prominence=prominence, distance=distance)

    if len(peaks) > 0:
        if len(peaks) > 2:
            intervals = np.diff(peaks)
            jitter = np.std(intervals) / np.mean(intervals)
        else:
            jitter = 0.0

        # The first peak after lag 0 represents the primary period
        confidence = corr[peaks[0]]

        return confidence > rtol and jitter < rtol

    return False


def determine_behavior(
    x: Array,
    *,
    n: int | None = None,
    fptol: float = 1.0e-3,
    lctol: float = 5.0e-2,
) -> Behavior:
    if n is None:
        n = int(0.1 * x.shape[0])

    if not np.isfinite(x[-n:]):
        return Behavior.Divergent

    # NOTE: for fixed point, we just look at the norm of the whole time series
    xd = np.linalg.norm(x[-n:], axis=1)
    if is_fixed_point(xd, xtol=fptol, gtol=fptol):
        return Behavior.FixedPoint

    # NOTE: for periodic checks, we first remove the mean, so that any fixed point
    # components become very small, and then take the norm.
    xd = np.linalg.norm(x - np.mean(x, axis=0), axis=1)
    if is_periodic(xd, rtol=lctol):
        return Behavior.Periodic

    return Behavior.Unknown
