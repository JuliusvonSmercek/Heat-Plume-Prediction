"""Render plot specs (``<stem>.npy`` + ``<stem>.yaml``) to images.

All matplotlib code lives here. Re-render after editing a spec yaml:

    python -m code.plotting results/.../rwpt [--format pdf] [--dpi 300]

A directory argument is searched recursively: every ``*.figure.yaml`` is rendered as a
multi-panel figure, every panel yaml not referenced by a figure is rendered on its own.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
from scipy.ndimage import maximum_filter  # noqa: E402

from .cmaps import register_jp_temperature_cmaps  # noqa: E402
from .spec import FIGURE_SUFFIX, FigureSpec, PlotSpec, load_panel, load_spec  # noqa: E402

log = logging.getLogger(__name__)


def _prepare_field(data: np.ndarray, spec: PlotSpec) -> tuple[np.ndarray, list[float] | None]:
    """Apply cutout and marker dilation (display only; the npy keeps raw values).

    Returns the field to draw and its imshow extent. A cutout keeps the axis values of
    the full field: the extent is cropped too, so ticks show the cut region's real
    coordinates (metres, or original pixel indices when no extent is set).
    """
    arr = np.asarray(data, dtype=np.float64)
    extent = spec.axes.extent
    if spec.cutout.enabled:
        n_rows, n_cols = arr.shape
        r0, r1, _ = slice(*spec.cutout.rows).indices(n_rows)
        c0, c1, _ = slice(*spec.cutout.cols).indices(n_cols)
        if r1 <= r0 or c1 <= c0:
            raise ValueError(f"Empty cutout rows={spec.cutout.rows} cols={spec.cutout.cols} for shape {arr.shape}")
        arr = arr[r0:r1, c0:c1]
        extent = _cutout_extent(extent, spec.axes.origin, (n_rows, n_cols), (r0, r1), (c0, c1))
    radius = int(spec.display.marker_dilation_px)
    if radius > 0:
        arr = maximum_filter(arr, size=2 * radius + 1)
    return arr, extent


def _cutout_extent(
    extent: list[float] | None,
    origin: str,
    shape: tuple[int, int],
    rows: tuple[int, int],
    cols: tuple[int, int],
) -> list[float]:
    """imshow extent ``[left, right, bottom, top]`` of ``[rows, cols]`` within the full field."""
    n_rows, n_cols = shape
    if extent is None:  # matplotlib's default: pixel centres at integer indices
        extent = [-0.5, n_cols - 0.5, n_rows - 0.5, -0.5] if origin == "upper" else [-0.5, n_cols - 0.5, -0.5, n_rows - 0.5]
    left, right, bottom, top = (float(v) for v in extent)
    x = lambda c: left + (right - left) * c / n_cols  # noqa: E731
    # Row 0 is drawn at ``top`` for origin="upper" and at ``bottom`` for origin="lower".
    if origin == "upper":
        y = lambda r: top + (bottom - top) * r / n_rows  # noqa: E731
        return [x(cols[0]), x(cols[1]), y(rows[1]), y(rows[0])]
    y = lambda r: bottom + (top - bottom) * r / n_rows  # noqa: E731
    return [x(cols[0]), x(cols[1]), y(rows[0]), y(rows[1])]


def _overlay_wells(ax, wells_rowcol_px: list[list[float]]) -> None:
    if not wells_rowcol_px:
        return
    wells = np.atleast_2d(np.asarray(wells_rowcol_px, dtype=float))
    ax.scatter(
        wells[:, 1],
        wells[:, 0],
        s=28,
        facecolors="none",
        edgecolors="k",
        linewidths=0.9,
        zorder=5,
    )


def _draw_panel(ax, data: np.ndarray, spec: PlotSpec) -> None:
    cm = spec.colormap
    if cm.jp_params:
        register_jp_temperature_cmaps(cm.jp_params["ambient_temperature_C"], cm.jp_params["temperature_spread_C"])

    axes = spec.axes
    field, extent = _prepare_field(data, spec)
    im = ax.imshow(
        field,
        cmap=cm.name,
        vmin=cm.vmin,
        vmax=cm.vmax,
        extent=extent,
        origin=axes.origin,
        interpolation=spec.display.interpolation,
    )
    ax.set_aspect(axes.aspect)
    _overlay_wells(ax, spec.overlays.wells_rowcol_px)

    if axes.show:
        ax.tick_params(axis="both", which="major", labelsize=axes.tick_labelsize)
        if axes.show_labels:
            ax.set_xlabel(axes.xlabel, fontsize=axes.tick_labelsize)
            ax.set_ylabel(axes.ylabel, fontsize=axes.tick_labelsize)
        if axes.show_title and spec.title:
            ax.set_title(spec.title, fontsize=axes.title_fontsize)
    else:
        ax.axis("off")

    if cm.colorbar:
        cax = make_axes_locatable(ax).append_axes("right", size=cm.colorbar_size, pad=0.05)
        cb = ax.figure.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=cm.colorbar_labelsize)
        if cm.colorbar_label:
            cb.set_label(cm.colorbar_label, fontsize=cm.colorbar_labelsize)


def _savefig(fig: Figure, out_path: Path, output, fmt: str | None, dpi: float | None) -> Path:
    fmt = fmt or output.format
    out_path = out_path.parent / f"{out_path.name}.{fmt}"
    kwargs = {"format": fmt, "dpi": dpi or output.dpi}
    if output.bbox_inches is not None:
        kwargs["bbox_inches"] = output.bbox_inches
    if output.pad_inches is not None:
        kwargs["pad_inches"] = output.pad_inches
    fig.savefig(out_path, **kwargs)
    return out_path


def _stem(yaml_path: Path) -> Path:
    name = yaml_path.name
    for suffix in (FIGURE_SUFFIX, ".yaml"):
        if name.endswith(suffix):
            return yaml_path.parent / name[: -len(suffix)]
    return yaml_path.parent / name


def render_panel(yaml_path: str | Path, fmt: str | None = None, dpi: float | None = None) -> Path:
    """Render one panel spec to ``<stem>.<format>`` next to the yaml."""
    yaml_path = Path(yaml_path)
    data, spec = load_panel(yaml_path)
    fig = Figure(figsize=tuple(spec.output.figsize))
    ax = fig.add_subplot(1, 1, 1) if spec.axes.show else fig.add_axes([0, 0, 1, 1])
    _draw_panel(ax, data, spec)
    if spec.axes.show:
        fig.set_layout_engine("tight")
    return _savefig(fig, _stem(yaml_path), spec.output, fmt, dpi)


def render_figure(yaml_path: str | Path, fmt: str | None = None, dpi: float | None = None) -> Path:
    """Render a ``*.figure.yaml`` grid of panels to ``<stem>.<format>``."""
    yaml_path = Path(yaml_path)
    figure = load_spec(yaml_path)
    if not isinstance(figure, FigureSpec):
        raise ValueError(f"{yaml_path}: expected a figure spec")
    panels = [load_panel(yaml_path.parent / name) for name in figure.panels]
    n_panels = len(panels)
    n_cols = min(figure.ncols, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    figsize = figure.output.figsize or [
        figure.panel_size[0] * n_cols,
        figure.panel_size[1] * n_rows + 0.4,
    ]
    fig = Figure(figsize=tuple(figsize))
    axes = fig.subplots(n_rows, n_cols, squeeze=False)

    for idx, (data, spec) in enumerate(panels):
        _draw_panel(axes[idx // n_cols][idx % n_cols], data, spec)
    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    if figure.suptitle:
        fig.suptitle(figure.suptitle, fontsize=figure.suptitle_fontsize)
    fig.set_layout_engine("tight")
    out_path = _savefig(fig, _stem(yaml_path), figure.output, fmt, dpi)
    log.info(f"Rendered {out_path}")
    return out_path


def render_path(path: str | Path, fmt: str | None = None, dpi: float | None = None) -> list[Path]:
    """Render a spec yaml, or every spec under a directory (panels of a figure only once)."""
    path = Path(path)
    if path.is_file():
        spec = load_spec(path)
        render = render_figure if isinstance(spec, FigureSpec) else render_panel
        return [render(path, fmt, dpi)]

    figures, panels = [], []
    for yaml_path in sorted(path.rglob("*.yaml")):
        try:
            spec = load_spec(yaml_path)
        except Exception:
            continue  # config.yaml, measurements.yaml, ...
        (figures if isinstance(spec, FigureSpec) else panels).append((yaml_path, spec))

    in_figure = {(p.parent / name).resolve() for p, fig in figures for name in fig.panels}
    written = [render_figure(p, fmt, dpi) for p, _ in figures]
    written += [render_panel(p, fmt, dpi) for p, _ in panels if p.resolve() not in in_figure]
    return written
