# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import enum
import pathlib
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from orbitkit.typing import Array, PathLike
from orbitkit.utils import BOOLEAN_STATES, module_logger, on_ci

if TYPE_CHECKING:
    import matplotlib.pyplot as mp

log = module_logger(__name__)


# {{{ utils


def _check_usetex(*, s: bool) -> bool:
    try:
        import matplotlib
    except ImportError:
        return False

    try:
        return bool(matplotlib.checkdep_usetex(s))  # ty: ignore[unresolved-attribute]
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
    if on_ci():
        return

    try:
        import matplotlib.pyplot as mp
    except ImportError:
        return

    # start off by resetting the defaults
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)

    import os

    if use_tex is None:
        use_tex = "GITHUB_REPOSITORY" not in os.environ and _check_usetex(s=True)

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
    vmin: float | None = -1.0,
    vmax: float | None = 1.0,
) -> tuple[str, ...]:
    from matplotlib import cm
    from matplotlib.colors import Normalize, to_hex

    if vmin is None:
        vmin = np.min(w)

    if vmax is None:
        vmax = np.max(w)

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
            ax.xaxis.pane.fill = pane_fill  # ty: ignore[unresolved-attribute]
            ax.yaxis.pane.fill = pane_fill  # ty: ignore[unresolved-attribute]
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
            mp.show(block=True)

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
    facecolor: str = "white",
    transparent: bool = False,
    overwrite: bool = True,
    **kwargs: Any,
) -> None:
    """A wrapper around :meth:`~matplotlib.figure.Figure.savefig`.

    :arg filename: a file name where to save the figure. If the file name does
        not have an extension, the default format from ``savefig.format`` is
        used.
    :arg normalize: if *True*, use :func:`slugify` to normalize the file name.
        Note that this will slugify any extensions as well and replace them
        with the default extension. If a certain extension is desired, it should
        probably be set in ``savefig.format``.
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
        raise FileExistsError(f"output file '{filename}' already exists")

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
        facecolor=facecolor,
        transparent=transparent,
        **kwargs,
    )


# }}}


# {{{ heatmap


def heatmap(
    ax: mp.Axes,
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
    ax: mp.Axes,
    t: Array,
    y: Array,
    *,
    height: float | None = None,
    distance: float | None = None,
    markerheight: float = 0.5,
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


# {{{ write_nx_from_adjacency


@enum.unique
class NetworkXLayout(enum.Enum):
    """Supported ``networkx``
    `layouts <https://networkx.org/documentation/stable/reference/drawing.html>`__.
    """

    Circular = enum.auto()
    """Position nodes on a circle."""
    ARF = enum.auto()
    """Attractive and Repulsive Forces (ARF) layout that improves the spring layout."""
    Bipartite = enum.auto()
    """Position nodes in two straight lines."""
    BFS = enum.auto()
    """Position nodes according to breadth-first search algorithm."""
    ForceAtlas2 = enum.auto()
    """Position nodes using the ForceAtlas2 force-directed layout algorithm."""
    KamadaKawai = enum.auto()
    """Position nodes using Kamada-Kawai path-length cost-function."""
    Planar = enum.auto()
    """Position nodes without edge intersections."""
    Shell = enum.auto()
    """Position nodes in concentric circles."""
    Spring = enum.auto()
    """Position nodes using Fruchterman-Reingold force-directed algorithm."""
    Spectral = enum.auto()
    """Position nodes using the eigenvectors of the graph Laplacian."""
    Spiral = enum.auto()
    """Position nodes in a spiral layout."""
    Multipartite = enum.auto()
    """Position nodes in layers of straight lines."""


def write_nx_from_adjacency(
    filename: PathLike,
    mat: Array,
    *,
    layout: NetworkXLayout = NetworkXLayout.ARF,
    overwrite: bool = False,
) -> None:
    import networkx as nx  # ty: ignore[unresolved-import,unused-ignore-comment,unused-ignore-comment]

    filename = pathlib.Path(filename)
    if not overwrite and filename.exists():
        raise FileExistsError(f"output file '{filename}' already exists")

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"'mat' must be a square matrix: {mat.shape}")

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"'mat' must be a square matrix: {mat.shape}")

    graph = nx.from_numpy_array(mat)
    if layout == NetworkXLayout.Circular:
        layout = nx.circular_layout(graph)
    elif layout == NetworkXLayout.ARF:
        layout = nx.arf_layout(graph)
    elif layout == NetworkXLayout.Bipartite:
        layout = nx.bipartite_layout(graph)
    elif layout == NetworkXLayout.BFS:
        layout = nx.bfs_layout(graph, graph.node(0))
    elif layout == NetworkXLayout.ForceAtlas2:
        layout = nx.forceatlas2_layout(graph)
    elif layout == NetworkXLayout.KamadaKawai:
        layout = nx.kamada_kawai_layout(graph)
    elif layout == NetworkXLayout.Planar:
        layout = nx.planar_layout(graph)
    elif layout == NetworkXLayout.Shell:
        layout = nx.shell_layout(graph)
    elif layout == NetworkXLayout.Spring:
        layout = nx.spring_layout(graph, iterations=1024)
    elif layout == NetworkXLayout.Spectral:
        layout = nx.spectral_layout(graph)
    elif layout == NetworkXLayout.Spiral:
        layout = nx.spiral_layout(graph)
    elif layout == NetworkXLayout.Multipartite:
        layout = nx.multipartite_layout(graph)
    else:
        raise ValueError(f"unsupported layout: {layout}")

    with figure(filename) as fig:
        ax = fig.gca()

        nx.draw_networkx_nodes(
            graph,
            layout,
            ax=ax,
        )
        nx.draw_networkx_labels(graph, layout, ax=ax)
        nx.draw_networkx_edges(
            graph,
            layout,
            ax=ax,
            arrows=True,
            connectionstyle="arc3,rad=0.1",
        )
        ax.set_axis_off()


# }}}


# {{{ dot


@enum.unique
class DotLayout(enum.Enum):
    """A list of `layout engines <https://graphviz.org/docs/layouts/>`__."""

    Neato = enum.auto()
    """Spring-type layout engine."""
    Dot = enum.auto()
    """Hierarchical or layered drawings of directed graphs."""
    FDP = enum.auto()
    """Force-Directed Placement layout engine."""
    SFDP = enum.auto()
    """A scalable FDP layout engine."""
    Circo = enum.auto()
    """A circular layout engine."""
    TwoPi = enum.auto()
    """A radial layout engine."""
    Osage = enum.auto()
    """An engine for clustered graphs."""
    Patchwork = enum.auto()
    """Draws map of clustered graph using a squarified treemap layout."""


def write_dot_from_adjacency(
    filename: PathLike,
    mat: Array,
    *,
    nodenames: Iterable[str] | None = None,
    nodecolors: Iterable[str] | None = None,
    layout: DotLayout = DotLayout.Neato,
    overwrite: bool = False,
) -> None:
    """Write a `*.dot* <https://graphviz.org/doc/info/lang.html>`__ file for the
    given adjacency matrix *mat*.

    :arg nodenames: a list of labels used for the nodes. Defaults to using
        the node index.
    :arg nodecolor: a list of colors for each node given as hexadeximal strings,
        i.e. ``#000000``. An alpha component can also be included.
    :arg layout: the layout used by dot to render the graph.
    """
    filename = pathlib.Path(filename)
    if not overwrite and filename.exists():
        raise FileExistsError(f"output file '{filename}' already exists")

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
        nodecolors = ("#000000",) * n
    else:
        nodecolors = tuple(nodecolors)

    if len(nodecolors) != n:
        raise ValueError(
            f"incorrect number of node colors: got {len(nodecolors)} but expected {n}"
        )

    with open(filename, "w", encoding="utf-8") as outf:
        outf.write("graph G {\n")
        outf.write(f"    layout={layout.name.lower()};\n")
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


# {{{ write_gexf_from_adjacency


def write_gexf_from_adjacency(
    filename: PathLike,
    mat: Array,
    *,
    directed: bool = False,
    nodenames: Iterable[str] | None = None,
    nodecolors: Iterable[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Write a `*.gexf* <https://gexf.net/>`__ file for the give adjacency
    matrix *mat*.

    :arg nodenames: a list of labels used for the nodes. Defaults to using
        the node index.
    """
    filename = pathlib.Path(filename)
    if not overwrite and filename.exists():
        raise FileExistsError(f"output file '{filename}' already exists")

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"'mat' must be a square matrix: {mat.shape}")

    n, _ = mat.shape
    if nodenames is None:
        nodenames = tuple(f"{i}" for i in range(n))
    else:
        nodenames = tuple(nodenames)

    if nodecolors is None:  # noqa: SIM108
        nodecolors = ("#000000",) * n
    else:
        nodecolors = tuple(nodecolors)

    edgedefault = "direct" if directed else "undirected"

    from xml.etree.ElementTree import (  # noqa: S405
        Element,
        ElementTree,
        SubElement,
        indent,
        register_namespace,
    )

    viz = "http://www.gexf.net/1.3/viz"
    register_namespace("viz", viz)

    # https://gexf.net/
    gexf = Element(
        "gexf",
        xmlns="http://www.gexf.net/1.3",
        version="1.3",
    )

    # {{{ metadata

    from datetime import datetime

    meta = SubElement(
        gexf,
        "meta",
        lastmodifieddate=datetime.now().strftime("%Y-%m-%d"),
    )
    creator = SubElement(meta, "creator")
    creator.text = "orbitkit"
    description = SubElement(meta, "description")
    description.text = f"{edgedefault} graph with {n} vertices"

    # }}}

    # {{{ graph

    from itertools import product

    graph = SubElement(
        gexf,
        "graph",
        mode="static",
        defaultedgetype=edgedefault,
    )

    nodes = SubElement(graph, "nodes")
    for i in range(n):
        node = SubElement(nodes, "node", id=str(i), label=nodenames[i])

        color = nodecolors[i][1:]
        r, g, b = (int(color[i : i + 2], 16) for i in (0, 2, 4))
        SubElement(node, f"{{{viz}}}color", r=str(r), g=str(g), b=str(b))

    edge_id = 0
    edges = SubElement(graph, "edges")
    for i, j in product(range(n), range(n)):
        if mat[i, j] == 0:
            continue

        if not directed and j <= i:
            continue

        SubElement(
            edges,
            "edge",
            edge_id=str(edge_id),
            source=str(i),
            target=str(j),
            weight=f"{mat[i, j]:g}",
        )
        edge_id += 1

    # }}}

    tree = ElementTree(gexf)
    indent(tree, space="  ", level=0)
    tree.write(filename, encoding="utf-8", xml_declaration=True)

    log.info("Saving '%s'", filename)


# }}}
