# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from orbitkit.typing import Array, PathLike, T

# {{{ environment


# fmt: off
BOOLEAN_STATES = {
    1: True, "1": True, "yes": True, "true": True, "on": True, "y": True,
    0: False, "0": False, "no": False, "false": False, "off": False, "n": False,
}
# fmt: on


def get_environ_boolean(name: str) -> bool:
    value = os.environ.get(name)
    return BOOLEAN_STATES.get(value.lower(), False) if value else False


# }}}


# {{{ logging


def module_logger(
    module: str,
    level: int | str | None = None,
) -> logging.Logger:
    """Create a new logging for the module *module*.

    The logger is created using a :class:`rich.logging.RichHandler` for fancy
    highlighting. The ``NO_COLOR`` environment variable can be used to
    disable colors.

    :arg module: a name for the module to create a logger for.
    :arg level: if *None*, the default value is taken to from the
        ``ORBITKIT_LOGGING_LEVEL`` environment variable and falls back to the
        ``INFO`` level if it does not exist (see :mod:`logging`).
    """
    if level is None:
        level = os.environ.get("ORBITKIT_LOGGING_LEVEL", "INFO").upper()

    if isinstance(level, str):
        level = getattr(logging, level.upper())

    assert isinstance(level, int)

    name, *rest = module.split(".", maxsplit=1)
    root = logging.getLogger(name)

    # FIXME: what is this??
    root.propagate = False

    if not root.handlers:
        from rich.highlighter import NullHighlighter
        from rich.logging import RichHandler

        no_color = "NO_COLOR" in os.environ
        handler = RichHandler(
            level,
            show_time=True,
            omit_repeated_times=False,
            show_level=True,
            show_path=True,
            highlighter=NullHighlighter() if no_color else None,
            markup=True,
        )

        root.addHandler(handler)

    root.setLevel(level)
    return root.getChild(rest[0]) if rest else root


log = module_logger(__name__)

# }}}


# {{{ matplotlib helpers


def check_usetex(*, s: bool) -> bool:
    try:
        import matplotlib
    except ImportError:
        return False

    try:
        return bool(matplotlib.checkdep_usetex(s))  # type: ignore[attr-defined]
    except AttributeError:
        # NOTE: simplified version from matplotlib
        # https://github.com/matplotlib/matplotlib/blob/ec85e725b4b117d2729c9c4f720f31cf8739211f/lib/matplotlib/__init__.py#L439=L456

        import shutil

        if not shutil.which("tex"):
            return False

        if not shutil.which("dvipng"):
            return False

        if not shutil.which("gs"):  # noqa: SIM103
            return False

        return True


def set_plotting_defaults(
    *,
    use_tex: bool | None = None,
    dark: bool | None = None,
    savefig_format: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Set custom :mod:`matplotlib` parameters.

    These are mainly used in the tests and examples to provide a uniform style
    to the results using `SciencePlots <https://github.com/garrettj403/SciencePlots>`__.
    For other applications, it is recommended to use local settings (e.g. in
    `matplotlibrc`).

    :arg use_tex: if *True*, LaTeX labels are enabled. By default, this checks
        if LaTeX is available on the system and only enables it if possible.
    :arg dark: if *True*, a dark default theme is selected instead of the
        default light one. If *None*, this takes its values from the ``ORBITKIT_DARK``
        boolean environment variable.
    :arg savefig_format: the format used when saving figures. By default, this
        uses the ``ORBITKIT_SAVEFIG`` environment variable and falls back to
        the :mod:`matplotlib` parameter ``savefig.format``.
    :arg overrides: a mapping of parameters to override the defaults. These
        can also be set separately after this function was called using ``rcParams``.
    """
    try:
        import matplotlib.pyplot as mp
    except ImportError:
        return

    # start off by resetting the defaults
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)

    import os

    if use_tex is None:
        use_tex = "GITHUB_REPOSITORY" not in os.environ and check_usetex(s=True)

    if not use_tex:
        log.warning("'use_tex' is disabled on this system.")

    if dark is None:
        tmp = os.environ.get("ORBITKIT_DARK", "off").lower()
        dark = BOOLEAN_STATES.get(tmp, False)

    if savefig_format is None:
        savefig_format = os.environ.get(
            "ORBITKIT_SAVEFIG", mp.rcParams["savefig.format"]
        ).lower()

    from contextlib import suppress

    # NOTE: preserve existing colors (the ones in "science" are ugly)
    prop_cycle = mp.rcParams["axes.prop_cycle"]
    with suppress(ImportError):
        import scienceplots  # noqa: F401

        mp.style.use(["science", "ieee"])

    # NOTE: the 'petroff10' style is available for version >= 3.10.0 and changes
    # the 'prop_cycle' to the 10 colors that are more accessible
    if "petroff10" in mp.style.available:
        mp.style.use("petroff10")
        prop_cycle = mp.rcParams["axes.prop_cycle"]

    defaults: dict[str, dict[str, Any]] = {
        "figure": {
            "figsize": (8, 8),
            "dpi": 300,
            "constrained_layout.use": True,
        },
        "savefig": {"format": savefig_format},
        "text": {"usetex": use_tex},
        "legend": {
            "fontsize": 20,
            "frameon": True,
            "fancybox": False,
            "edgecolor": "black",
        },
        "lines": {"linewidth": 2, "markersize": 10},
        "axes": {
            "labelsize": 28,
            "titlesize": 28,
            "grid": True,
            "grid.axis": "both",
            "grid.which": "both",
            "prop_cycle": prop_cycle,
        },
        "xtick": {"labelsize": 20, "direction": "in"},
        "ytick": {"labelsize": 20, "direction": "in"},
        "xtick.major": {"size": 6.5, "width": 1.5},
        "ytick.major": {"size": 6.5, "width": 1.5},
        "xtick.minor": {"size": 4.0},
        "ytick.minor": {"size": 4.0},
    }

    if dark:
        # NOTE: this is the black color used by the sphinx-book theme
        black = "111111"
        gray = "28313D"
        defaults["text"].update({"color": "white"})
        defaults["axes"].update({
            "labelcolor": "white",
            "facecolor": gray,
            "edgecolor": "white",
        })
        defaults["xtick"].update({"color": "white"})
        defaults["ytick"].update({"color": "white"})
        defaults["figure"].update({"facecolor": black, "edgecolor": black})
        defaults["savefig"].update({"facecolor": black, "edgecolor": black})

    for group, params in defaults.items():
        mp.rc(group, **params)

    if overrides:
        for group, params in overrides.items():
            mp.rc(group, **params)


# }}}


# {{{ Estimated Order of Convergence (EOC)


@dataclass(frozen=True)
class EOCRecorder:
    """Keep track of all your *estimated order of convergence* needs."""

    name: str = "Error"
    """A string identifier for the value which is estimated."""
    order: float | None = None
    """An expected order of convergence, if any."""

    history: list[tuple[float, float]] = field(default_factory=list, repr=False)
    """A list of ``(h, error)`` entries added from :meth:`add_data_point`."""

    @classmethod
    def from_data(
        cls, name: str, h: Array, error: Array, *, order: float | None = None
    ) -> EOCRecorder:
        eoc = cls(name=name, order=order)
        for i in range(h.size):
            eoc.add_data_point(h[i], error[i])

        return eoc

    def add_data_points(self, h: Array, error: Array) -> None:
        """Add multiple data points using :meth:`add_data_point`."""
        for h_i, e_i in zip(h, error, strict=True):
            self.add_data_point(h_i, e_i)

    def add_data_point(self, h: Any, error: Any) -> None:
        """Add a data point to the estimation.

        Note that both *h* and *error* need to be convertible to a float.

        :arg h: abscissa, a value representative of the "grid size".
        :arg error: error at given *h*.
        """
        self.history.append((float(h), float(error)))

    @property
    def estimated_order(self) -> float:
        """Estimated order of convergence for currently available data. The
        order is estimated by least squares through the given data
        (see :func:`estimate_order_of_convergence`).
        """
        if not self.history:
            return np.nan

        h, error = np.array(self.history).T
        _, eoc = estimate_order_of_convergence(h, error)
        return eoc

    @property
    def max_error(self) -> float:
        """Largest error (in absolute value) in current data."""
        r = np.amax(np.array([error for _, error in self.history]))
        return float(r)

    def __str__(self) -> str:
        return stringify_eoc(self)


def estimate_order_of_convergence(x: Array, y: Array) -> tuple[float, float]:
    """Computes an estimate of the order of convergence in the least-square sense.
    This assumes that the :math:`(x, y)` pair follows a law of the form

    .. math::

        y = m x^p

    and estimates the constant :math:`m` and power :math:`p`.
    """
    assert x.size == y.size
    if x.size <= 1:
        raise RuntimeError("Need at least two values to estimate order.")

    eps = np.finfo(x.dtype).eps
    c = np.polyfit(np.log10(x + eps), np.log10(y + eps), 1)
    return 10 ** c[-1], c[-2]


def estimate_gliding_order_of_convergence(
    x: Array, y: Array, *, gliding_mean: int | None = None
) -> Array:
    assert x.size == y.size
    if x.size <= 1:
        raise RuntimeError("Need at least two values to estimate order.")

    if gliding_mean is None:
        gliding_mean = x.size

    npoints = x.size - gliding_mean + 1
    return np.array(
        [
            estimate_order_of_convergence(
                x[i : i + gliding_mean], y[i : i + gliding_mean] + 1.0e-16
            )
            for i in range(npoints)
        ],
        dtype=x.dtype,
    )


def flatten(iterable: Iterable[Iterable[T]]) -> tuple[T, ...]:
    from itertools import chain

    return tuple(chain.from_iterable(iterable))


def stringify_eoc(*eocs: EOCRecorder) -> str:
    r"""
    :arg eocs: an iterable of :class:`EOCRecorder`\ s that are assumed to have
        the same number of entries in their histories.
    :returns: a string representing the results in *eocs* in the
        GitHub Markdown format.
    """
    histories = [np.array(eoc.history).T for eoc in eocs]
    orders = [
        estimate_gliding_order_of_convergence(h, error, gliding_mean=2)
        for h, error in histories
    ]

    h = histories[0][0]
    ncolumns = 1 + 2 * len(eocs)
    nrows = h.size

    lines = []
    lines.append(("h", *flatten([(eoc.name, "EOC") for eoc in eocs])))

    lines.append((":-:",) * ncolumns)

    for i in range(nrows):
        values = flatten([
            (
                f"{error[i]:.6e}",
                "---" if i == 0 else f"{order[i - 1, 1]:.3f}",
            )
            for (_, error), order in zip(histories, orders, strict=True)
        ])
        lines.append((f"{h[i]:.3e}", *values))

    lines.append((
        "Overall",
        *flatten([("", f"{eoc.estimated_order:.3f}") for eoc in eocs]),
    ))

    if any(eoc.order is not None for eoc in eocs):
        expected = flatten([
            (("", f"{eoc.order:.3f}") if eoc.order is not None else ("", "--"))
            for eoc in eocs
        ])

        lines.append(("Expected", *expected))

    widths = [max(len(line[i]) for line in lines) for i in range(ncolumns)]
    formats = ["{:%s}" % w for w in widths]  # noqa: UP031

    return "\n".join([
        " | ".join(fmt.format(value) for fmt, value in zip(formats, line, strict=True))
        for line in lines
    ])


def visualize_eoc(
    filename: PathLike,
    *eocs: EOCRecorder,
    order: float | None = None,
    abscissa: str | Literal[False] = "h",
    ylabel: str | Literal[False] = "Error",
    olabel: str | Literal[False] | None = None,
    enable_legend: bool = True,
    overwrite: bool = True,
) -> None:
    """Plot the given :class:`EOCRecorder` instances in a loglog plot.

    :arg filename: output file name for the figure.
    :arg order: expected order for all the errors recorded in *eocs*.
    :arg abscissa: name for the abscissa.
    """
    if not eocs:
        raise ValueError("no EOCRecorders are provided")

    if order is not None and order <= 0.0:
        raise ValueError(f"The 'order' should be a non-negative real number: {order}")

    from orbitkit.visualization import figure

    markers = ["o", "v", "^", "<", ">", "x", "+", "d", "D"][: len(eocs)]
    with figure(filename, overwrite=overwrite) as fig:
        ax = fig.gca()

        # {{{ plot eocs

        eocs_have_order = False
        extent = (-np.inf, np.inf, -np.inf)
        line = None
        for eoc, marker in zip(eocs, markers, strict=True):
            h, error = np.array(eoc.history).T
            ax.loglog(h, error, marker=marker, label=eoc.name)

            imax = np.argmax(h)
            max_h = h[imax]
            max_e = error[imax]
            min_e = np.min(error)
            extent = (
                max(extent[0], max_h),
                min(extent[1], min_e),
                max(extent[2], max_e),
            )

            if eoc.order is not None:
                eocs_have_order = True
                order = eoc.order
                min_h = np.exp(np.log(max_h) + np.log(min_e / max_e) / eoc.order)
                (line,) = ax.loglog(
                    [max_h, min_h],
                    [max_e, min_e],
                    "k--",
                )

        if abscissa and line is not None:
            if olabel is None:
                hname = abscissa.strip("$")
                if order == 1:
                    olabel = f"$O({hname})$"
                else:
                    olabel = f"$O({hname}^{{{order:g}}})$"

            if olabel:
                line.set_label(olabel)

        # }}}

        # {{{ plot order

        if order is not None and not eocs_have_order:
            max_h, min_e, max_e = extent

            min_h = np.exp(np.log(max_h) + np.log(min_e / max_e) / order)
            (line,) = ax.loglog(
                [max_h, min_h],
                [max_e, min_e],
                "k--",
            )

            if olabel:
                line.set_label(olabel)

            ax.tick_params(axis="x", which="both", rotation=45)

        # }}}

        ax.grid(visible=True, which="major", linestyle="-", alpha=0.75)
        ax.grid(visible=True, which="minor", linestyle="--", alpha=0.5)

        if abscissa:
            ax.set_xlabel(abscissa)

        if ylabel:
            ax.set_ylabel(ylabel)

        if enable_legend and (len(eocs) > 1 or (line and olabel)):
            ax.legend()


# }}}


# {{{ TicTocTimer


@dataclass
class TicTocTimer:
    """A simple timer that tries to copy MATLAB's ``tic`` and ``toc`` functions.

    .. code:: python

        time = TicTocTimer()
        time.tic()

        # ... do some work ...

        elapsed = time.toc()
        print(time)
    """

    t_wall_start: float = field(default=0.0, init=False)
    t_wall: float = field(default=0.0, init=False)

    n_calls: int = field(default=0, init=False)
    t_avg: float = field(default=0.0, init=False)
    t_sqr: float = field(default=0.0, init=False)

    def tic(self) -> None:
        """Start the timer."""
        self.t_wall = 0.0
        self.t_wall_start = time.perf_counter()

    def toc(self) -> float:
        """Stop the timer and update internal statistics."""
        self.t_wall = time.perf_counter() - self.t_wall_start

        # statistics
        self.n_calls += 1

        delta0 = self.t_wall - self.t_avg
        self.t_avg += delta0 / self.n_calls
        delta1 = self.t_wall - self.t_avg
        self.t_sqr += delta0 * delta1

        return self.t_wall

    def __str__(self) -> str:
        # NOTE: this matches how MATLAB shows the time from `toc`.
        return f"Elapsed time is {self.t_wall:.5f} seconds."

    def stats(self) -> str:
        """Aggregate statistics across multiple calls to :meth:`toc`."""
        # NOTE: n_calls == 0 => toc was not called yet, so stddev is zero
        #       n_calls == 1 => only one call to toc, so the stddev is zero
        t_std = np.sqrt(self.t_sqr / (self.n_calls - 1)) if self.n_calls > 1 else 0.0

        return f"avg {self.t_avg:.3f}s ± {t_std:.3f}s"

    def short(self) -> str:
        """A shorter string for the last :meth:`tic`-:meth:`toc` cycle."""
        return f"wall {self.t_wall:.5f}s"


# }}}


# {{{ timeit


@dataclass(frozen=True)
class TimingResult:
    """Statistics for a set of runs (see :func:`timeit`)."""

    walltime: float
    """Minimum walltime for a set of runs."""
    mean: float
    """Mean walltime for a set of runs."""
    std: float
    """Standard derivation for a set of runs."""

    @classmethod
    def from_results(cls, results: list[float]) -> TimingResult:
        """Gather statistics from a set of runs."""
        rs = np.array(results)

        return TimingResult(
            walltime=np.min(rs),
            mean=np.mean(rs),
            std=np.std(rs, ddof=1),
        )

    def __str__(self) -> str:
        return f"{self.mean:.5f}s ± {self.std:.3f}"


def timeit(
    stmt: Callable[[], Any],
    *,
    repeat: int = 1000,
    number: int = 10,
    skip: int = 0,
) -> TimingResult:
    """Run *stmt* using :func:`timeit.repeat`.

    :arg repeat: number of times to call :func:`timeit.timeit` (inside of
        :func:`timeit.repeat`).
    :arg number: number of times to run the *stmt* in each call to
        :func:`timeit.timeit`.
    :arg skip: number of leading calls from *repeat* to skip, e.g. to
        avoid measuring an initial cache hit.
    :returns: a :class:`TimingResult` with statistics about the runs.
    """

    import timeit as _timeit

    r = _timeit.repeat(stmt=stmt, repeat=repeat + 1, number=number)
    return TimingResult.from_results(r[skip:])


# }}}
