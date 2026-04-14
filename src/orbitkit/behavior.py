# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from orbitkit.typing import Array1D, Array2D
from orbitkit.utils import module_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib.pyplot as mp
    from matplotlib.cm import ScalarMappable
    from matplotlib.contour import ContourSet

log = module_logger(__name__)


# {{{ determine_behavior


@enum.unique
class Behavior(enum.IntEnum):
    """Macro view of the behavior of a dynamical system."""

    Unknown = 0
    """System has an unknown behavior that could not be determined."""
    Divergent = 1
    """System has diverged to infinity."""

    Chaotic = 2
    """System is chaotic, i.e. it has a (sufficiently) positive Lyapunov exponent."""
    FixedPoint = 4
    """System has reached a fixed point (or steady state)."""
    Periodic = 8
    """System has reached a cycle, i.e. all components are periodic, but not
    necessarily synchronized.
    """


BEHAVIOR_FULL_NAME = {
    Behavior.Unknown: "Unknown",
    Behavior.Divergent: "Divergent",
    Behavior.Chaotic: "Chaotic",
    Behavior.FixedPoint: "Fixed Point",
    Behavior.Periodic: "Periodic",
}


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

    log.debug(
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


# }}}

# {{{ visualize_behavior_probability


def visualize_behavior_probability(
    ax: mp.Axes,
    x: Array1D[np.floating[Any]],
    y: Array1D[np.floating[Any]],
    z: Array2D[np.floating[Any]],
    *,
    cmap: str = "seismic",
) -> ScalarMappable:
    from orbitkit.visualization import heatmap

    im = heatmap(ax, x, y, z, cmap=cmap, vmin=0.0, vmax=1.0)

    return im


# }}}

# {{{ visualize_behavior_probability_entropy


def visualize_behavior_probability_entropy(
    ax: mp.Axes,
    x: Array1D[np.floating[Any]],
    y: Array1D[np.floating[Any]],
    zs: dict[Behavior, Array2D[np.floating[Any]]],
    *,
    cmap: str = "Blues",
    contours: bool = False,
    level: float = 0.95,
    linewidth: float | None = None,
    linecolor: str = "w",
    base: int = 2,
) -> ScalarMappable:
    """Visualize the entropy in the systems behaviors in *zs*.

    The dictionary *zs* contains a mapping from behaviors to their probability
    in :math:`[0, 1]` for each of the parameter pairs ``(x[i], y[j])``.
    """
    from scipy.stats import entropy

    Z = np.stack(list(zs.values()), axis=-1)
    residual = np.clip(1.0 - Z.sum(axis=-1, keepdims=True), 0, 1)
    Z = np.concatenate([Z, residual], axis=-1)
    H = entropy(Z, axis=-1, base=base)

    from orbitkit.visualization import heatmap

    im = heatmap(
        ax,
        x,
        y,
        H,
        cmap="Blues",
        linecolor=linecolor,
        xlinewidth=linewidth,
        ylinewidth=linewidth,
    )

    if contours:
        visualize_behavior_probability_contour(
            ax, x, y, zs, level=level, linewidth=linewidth
        )

    return im


# }}}


# {{{ visualize_behavior_probability_contour


def visualize_behavior_probability_contour(
    ax: mp.Axes,
    x: Array1D[np.floating[Any]],
    y: Array1D[np.floating[Any]],
    zs: dict[Behavior, Array2D[np.floating[Any]]],
    *,
    level: float = 0.95,
    linewidth: float | None = None,
) -> Sequence[ContourSet]:
    """Visualize the contours in the systems behaviors in *zs*.

    The dictionary *zs* contains a mapping from behaviors to their probability
    in :math:`[0, 1]` for each of the parameter pairs ``(x[i], y[j])``.
    """
    from orbitkit.visualization import get_color_cycle

    X, Y = np.meshgrid(x, y)
    colors = get_color_cycle()

    contours = []
    for i, z in enumerate(zs.values()):
        c = ax.contour(
            X,
            Y,
            z,
            levels=[level],
            colors=colors[i],
            linestyles="-",
            linewidths=linewidth,
        )
        contours.append(c)

    return contours


# }}}


# {{{ visualize_behavior_probability_phase


def visualize_behavior_probability_phase(
    ax: mp.Axes,
    x: Array1D[np.floating[Any]],
    y: Array1D[np.floating[Any]],
    zs: dict[Behavior, Array2D[np.floating[Any]]],
    *,
    cmap: str = "Blues",
) -> ScalarMappable:
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

    from orbitkit.visualization import get_rgb_color_cycle, heatmap

    colors = get_rgb_color_cycle()
    hues = {
        btype: rgb_to_hsv(color)[0] for btype, color in zip(zs, colors, strict=False)
    }

    # compute confidence
    Z = np.stack(list(zs.values()), axis=-1)  # (ny, nx, K)
    Zidx = Z.argmax(axis=-1)
    confidence = Z.max(axis=-1)

    ny, nx = Zidx.shape
    hsv = np.zeros((ny, nx, 3))
    for i, btype in enumerate(zs):
        mask = Zidx == i
        hsv[mask, 0] = hues[btype]
    hsv[:, :, 1] = confidence**1.25
    hsv[:, :, 2] = 0.88

    # plot
    rgb = hsv_to_rgb(hsv)
    im = heatmap(ax, x, y, rgb, linecolor="k", tickdensity=1.0)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(
                facecolor=hsv_to_rgb([hues[btype], 0.85, 0.88]),  # ty: ignore[invalid-argument-type]
                label=BEHAVIOR_FULL_NAME[btype],
            )
            for btype in zs
        ],
        loc="upper right",
    )

    return im


# }}}
