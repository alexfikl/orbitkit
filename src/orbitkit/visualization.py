# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from orbitkit.typing import Array, PathLike
from orbitkit.utils import module_logger

log = module_logger(__name__)


if TYPE_CHECKING:
    import matplotlib.pyplot as mp

# {{{ utils


def slugify(stem: str, separator: str = "_") -> str:
    """
    :returns: an ASCII slug representing *stem*, with all the unicode cleaned up
        and all non-standard separators replaced.
    """
    import re
    import unicodedata

    stem = unicodedata.normalize("NFKD", stem)
    stem = stem.encode("ascii", "ignore").decode().lower()
    stem = re.sub(r"[^a-z0-9]+", separator, stem)
    stem = re.sub(rf"[{separator}]+", separator, stem.strip(separator))

    return stem


def to_color(
    w: Array,
    *,
    colormap: str = "turbo",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> tuple[str, ...]:
    from matplotlib import cm
    from matplotlib.colors import Normalize, to_hex

    cmap = cm.get_cmap(colormap)
    norm = Normalize(vmin=vmin, vmax=vmax)

    colors = cmap(norm(w))
    return tuple(to_hex(color).upper() for color in colors)


# }}}


# {{{ figure context manager


@contextmanager
def figure(
    filename: PathLike | None = None,
    nrows: int = 1,
    ncols: int = 1,
    *,
    pane_fill: bool = False,
    projection: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> Iterator[Any]:
    """A small wrapper context manager around :class:`matplotlib.figure.Figure`.

    :arg nrows: number of rows of subplots.
    :arg ncols: number of columns of subplots.
    :arg projection: a projection for all the axes in this figure, see
        :mod:`matplotlib.projections`.
    :arg figsize: the size of the resulting figure, set to
        ``(L * ncols, L * nrows)`` by default.
    :arg kwargs: Additional arguments passed to :func:`savefig`.
    :returns: the :class:`~matplotlib.figure.Figure` that was constructed. On exit
        from the context manager, the figure is saved to *filename* and closed.
    """
    import matplotlib.pyplot as mp

    fig = mp.figure()
    for i in range(nrows * ncols):
        fig.add_subplot(nrows, ncols, i + 1, projection=projection)

    # FIXME: get size of one figure
    if figsize is None:
        width, height = mp.rcParams["figure.figsize"]
        figsize = (width * ncols, height * nrows)
    fig.set_size_inches(*figsize)

    if projection == "3d":
        from mpl_toolkits.mplot3d.axes3d import Axes3D

        for ax in fig.axes:
            assert isinstance(ax, Axes3D)
            ax.xaxis.pane.fill = pane_fill
            ax.yaxis.pane.fill = pane_fill
            ax.zaxis.pane.fill = pane_fill

    try:
        yield fig
    finally:
        if projection == "3d":
            for ax in fig.axes:
                assert isinstance(ax, Axes3D)
                ax.set_box_aspect((4, 4, 4), zoom=1.1)

        if filename is not None:
            savefig(fig, filename, **kwargs)
        else:
            mp.show(block=True)  # type: ignore[no-untyped-call,unused-ignore]

        mp.close(fig)


# }}}


# {{{ savefig wrapper


def savefig(
    fig: Any,
    filename: PathLike,
    *,
    bbox_inches: str = "tight",
    pad_inches: float = 0,
    normalize: bool = False,
    overwrite: bool = True,
    **kwargs: Any,
) -> None:
    """A wrapper around :meth:`~matplotlib.figure.Figure.savefig`.

    :arg filename: a file name where to save the figure. If the file name does
        not have an extension, the default format from ``savefig.format`` is
        used.
    :arg normalize: if *True*, use :func:`slugify` to normalize the file name.
        Note that this will slugify any extensions as well and replace them
        with the default extension.
    :arg overwrite: if *True*, any existing files are overwritten.
    :arg kwargs: renaming arguments are passed directly to ``savefig``.
    """
    import matplotlib.pyplot as mp

    ext = mp.rcParams["savefig.format"]
    filename = pathlib.Path(filename)

    if normalize:
        # NOTE: slugify(name) will clubber any prefixes, so we special-case a
        # few of them here to help out the caller
        if filename.suffix in {".png", ".jpg", ".jpeg", ".pdf", ".eps", ".tiff"}:
            filename = filename.with_stem(slugify(filename.stem))
        else:
            filename = filename.with_name(slugify(filename.name)).with_suffix(f".{ext}")

    if not filename.suffix:
        filename = filename.with_suffix(f".{ext}").resolve()

    if not overwrite and filename.exists():
        raise FileExistsError(f"Output file '{filename}' already exists")

    bbox_extra_artists = []
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            bbox_extra_artists.append(legend)

    log.info("Saving '%s'", filename)
    fig.savefig(
        filename,
        bbox_extra_artists=tuple(bbox_extra_artists),
        bbox_inches="tight",
        pad_inches=pad_inches,
        **kwargs,
    )


# }}}


# {{{ heatmap


def heatmap(
    ax: mp.Axes,  # type: ignore[name-defined]
    x: Array,
    y: Array,
    z: Array,
    *,
    title: str | None = None,
    cmap: str = "jet",
    alpha: float | Array | None = None,
    vmax: float = 1.0,
    shrink: float = 0.7,
    linecolor: str = "w",
    linewidth: float = 1.0,
    xrotation: float = 45.0,
) -> Any:
    """Plot a heatmap for a given array.

    This is just a :func:`~matplotlib.pyplot.imshow` plot with some specific
    commands. In particular, it overlays a custom grid on top of the image
    that exactly corresponds to the "pixels" in *z*.
    """
    # make uniform grid for display
    xs = np.linspace(x[0], x[-1], x.size)
    ys = np.linspace(y[0], y[-1], y.size)

    # define extent
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    extent = (xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dy / 2, ys[-1] + dy / 2)

    im = ax.imshow(
        z,
        extent=extent,
        interpolation="none",
        aspect="auto",
        cmap=cmap,
        alpha=alpha,
        vmin=0.0,
        vmax=vmax,
        origin="lower",
    )

    indices = np.linspace(0, x.size - 1, 10, endpoint=True, dtype=np.int32)
    ax.set_xticks(xs[indices], [f"{xi:.2f}" for xi in x[indices]])
    indices = np.linspace(0, y.size - 1, 10, endpoint=True, dtype=np.int32)
    ax.set_yticks(ys[indices], [f"{yi:.2f}" for yi in y[indices]])
    ax.tick_params(which="minor", length=0)

    ax.set_box_aspect(1)
    ax.grid(visible=False, which="both")
    ax.tick_params(axis="x", rotation=xrotation)

    if linewidth > 0.0:
        for j in range(xs.size - 1):
            ax.axvline(xs[j] + 0.5 * dx, color=linecolor, lw=linewidth)

        for j in range(ys.size - 1):
            ax.axhline(ys[j] + 0.5 * dy, color=linecolor, lw=linewidth)

    return im


# }}}


# {{{ rastergram


def rastergram(
    ax: mp.Axes,  # type: ignore[name-defined]
    t: Array,
    y: Array,
    *,
    height: float | None = None,
    distance: float | None = None,
    markerheight: float | None = 0.5,
    markerwidth: float | None = None,
) -> None:
    """Plot the rastergram for the given signal.

    This is a simple wrapper around :func:`matplotlib.pyplot.eventplot`.

    :arg height: required height of the peaks (see :func:`scipy.signal.find_peaks`).
    :arg distance: required minimal horizontal distance between the peaks
        (see :func:`scipy.signal.find_peaks`).
    """
    # {{{ find peaks

    from scipy.signal import find_peaks

    peaks = []
    for i in range(y.shape[0]):
        peaks_i, _ = find_peaks(y[i], height=height, distance=distance)
        peaks.append(t[peaks_i])

    # }}}

    # {{{ estimate linewidths and linelengths

    if markerwidth is None and markerwidth is not None:
        fig = ax.get_figure()
        figwidth, figheight = fig.get_size_inches()
        _, _, wfrac, _ = ax.get_position().bounds

        xmin, xmax = t[0], t[-1]
        ymin, ymax = -0.5, y.shape[0] - 0.5

        frac = figheight / figwidth / wfrac
        markerwidth = frac * markerheight * (xmax - xmin) / (ymax - ymin)

    # }}}

    ax.eventplot(peaks, linelengths=markerheight, linewidths=markerwidth, color="black")


# }}}


# {{{ dot


def write_dot_from_adjacency(
    filename: PathLike,
    mat: Array,
    *,
    nodenames: Iterable[str] | None = None,
    nodecolors: Iterable[str] | None = None,
    layout: str = "neato",
    overwrite: bool = False,
) -> None:
    filename = pathlib.Path(filename)
    if not overwrite and filename.exists():
        raise FileExistsError(f"Output file '{filename}' already exists")

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"'mat' must be a square matrix: {mat.shape}")

    n, _ = mat.shape
    if nodenames is None:
        nodenames = tuple(f"{i}" for i in range(n))
    else:
        nodenames = tuple(nodenames)

    if len(nodenames) != n:
        raise ValueError(
            f"incorrect number of node names: got {len(nodenames)} but expected {n}"
        )

    if nodecolors is None:  # noqa: SIM108
        nodecolors = ("white",) * n
    else:
        nodecolors = tuple(nodecolors)

    if len(nodecolors) != n:
        raise ValueError(
            f"incorrect number of node colors: got {len(nodecolors)} but expected {n}"
        )

    with open(filename, "w", encoding="utf-8") as outf:
        outf.write("graph G {\n")
        outf.write(f"    layout={layout};\n")
        outf.write("    overlap=false;\n")
        outf.write('    sep="+0.5";\n')
        outf.write("\n")
        outf.write("    node [penwidth=2, shape=circle, width=1.0];\n")
        outf.write('    edge [penwidth=10, color="#6782A7C0"];\n')
        outf.write("\n")

        for i, (name, color) in enumerate(zip(nodenames, nodecolors, strict=True)):
            rgba = color
            if len(rgba) == 7:
                rgba = f"{color}C0"
            assert len(rgba) == 9

            outf.write(
                f'    "{i}" [label="{name}", style=filled, fillcolor="{rgba}"];\n'
            )

        outf.write("\n")

        for i in range(n):
            for j in range(i + 1, n):
                if mat[i, j] != 0:
                    outf.write(f'    "{i}" -- "{j}";\n')

        outf.write("}\n")

    log.info("Saving '%s'", filename)


# }}}
