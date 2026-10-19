"""Plot specifications: what to show, decoupled from how it is drawn.

Every plot is stored as ``<stem>.npy`` (raw 2-D values) plus ``<stem>.yaml`` (a
``PlotSpec``: colormap, limits, extent, axis labels, cutout, overlays, output format).
Multi-panel figures additionally get ``<stem>.figure.yaml`` listing their panel yamls.

Only numpy and yaml are needed here; ``render`` adds matplotlib.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Derived panels: ``data: {op: abs_diff, of: [a.npy, b.npy]}`` is computed on load, no npy of its own.
DERIVED_OPS = {
    "abs_diff": lambda a, b: np.abs(a - b),
    "diff": lambda a, b: a - b,
}

PANEL_KIND = "panel"
FIGURE_KIND = "figure"
FIGURE_SUFFIX = ".figure.yaml"

@dataclass
class ColormapSpec:
    name: str = "jp_linear"
    vmin: float | None = None
    vmax: float | None = None
    # ambient / spread used to (re-)register the jp_temperature_* colormaps
    jp_params: dict[str, float] | None = None
    colorbar: bool = True
    colorbar_label: str | None = None
    colorbar_labelsize: float = 20
    colorbar_size: float | str = 0.3  # inches (float) or percent of the axes ("4%")


@dataclass
class AxesSpec:
    show: bool = True  # False -> ax.axis("off")
    show_labels: bool = False
    show_title: bool = False
    xlabel: str = "y [m]"
    ylabel: str = "x [m]"
    # imshow extent (left, right, bottom, top); None -> pixel indices
    extent: list[float] | None = None
    origin: str = "upper"
    aspect: str | float = "equal"
    tick_labelsize: float = 20
    title_fontsize: float = 16


@dataclass
class CutoutSpec:
    enabled: bool = False
    rows: list[int] = field(default_factory=lambda: [100, 400])
    cols: list[int] = field(default_factory=lambda: [100, 400])


@dataclass
class DisplaySpec:
    interpolation: str = "nearest"
    marker_dilation_px: int = 0  # >0 widens single-cell markers (heat pump positions)


@dataclass
class OverlaySpec:
    wells_rowcol_px: list[list[float]] = field(default_factory=list)


@dataclass
class OutputSpec:
    format: str = "png"
    dpi: float = 160
    figsize: list[float] = field(default_factory=lambda: [8.4, 6.5])
    bbox_inches: str | None = "tight"
    pad_inches: float | None = None


@dataclass
class PlotSpec:
    data: str | dict[str, Any] = ""  # npy file name relative to the yaml, or a derived {op, of: [npy, ...]}
    kind: str = PANEL_KIND
    title: str | None = None
    quantity: str | None = None  # physical property shown
    colormap: ColormapSpec = field(default_factory=ColormapSpec)
    axes: AxesSpec = field(default_factory=AxesSpec)
    cutout: CutoutSpec = field(default_factory=CutoutSpec)
    display: DisplaySpec = field(default_factory=DisplaySpec)
    overlays: OverlaySpec = field(default_factory=OverlaySpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    meta: dict[str, Any] = field(default_factory=dict)  # free-form info (e.g. metrics); not drawn


@dataclass
class FigureSpec:
    panels: list[str] = field(default_factory=list)  # panel yaml names, relative to this yaml
    kind: str = FIGURE_KIND
    suptitle: str | None = None
    suptitle_fontsize: float = 12
    ncols: int = 2
    panel_size: list[float] = field(default_factory=lambda: [5.8, 4.6])
    output: OutputSpec = field(default_factory=lambda: OutputSpec(dpi=140, figsize=[]))
    meta: dict[str, Any] = field(default_factory=dict)  # free-form info (e.g. metrics); not drawn


# --- (de)serialization ---


def _plain(value: Any) -> Any:
    """Convert numpy scalars/arrays and tuples to YAML-safe Python types."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _from_dict(cls, data: dict[str, Any]):
    """Build a (nested) spec dataclass from a dict; unknown keys raise to catch typos."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else None
        if is_dataclass(default) and isinstance(value, dict):
            value = _from_dict(type(default), value)
        kwargs[name] = value
    return cls(**kwargs)


def _dump_yaml(spec, path: Path) -> None:
    path.write_text(yaml.safe_dump(_plain(asdict(spec)), sort_keys=False, allow_unicode=True), encoding="utf-8")


def _with_suffix(stem: Path, suffix: str) -> Path:
    # Stems may contain dots or brackets (e.g. "Temperature [C]"), so never use Path.with_suffix.
    return stem.parent / f"{stem.name}{suffix}"


def _as_2d(data) -> np.ndarray:
    array = np.squeeze(np.asarray(data, dtype=np.float32))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D field after squeeze, got shape {array.shape}")
    return array


def derived(op: str, *npy_names: str) -> dict[str, Any]:
    """``PlotSpec.data`` for a panel computed from other panels' npy files."""
    if op not in DERIVED_OPS:
        raise ValueError(f"Unknown derived op {op!r}; choose from {sorted(DERIVED_OPS)}")
    return {"op": op, "of": list(npy_names)}


def write_panel(path_stem: str | Path, data: np.ndarray | None, spec: PlotSpec) -> Path:
    """Write ``<stem>.npy`` + ``<stem>.yaml``; returns the yaml path.

    With ``data=None`` the spec must be derived (see ``derived``) and only the yaml is written.
    """
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    if data is None:
        if not isinstance(spec.data, dict):
            raise ValueError(f"{path_stem}: data=None needs a derived spec.data, got {spec.data!r}")
    else:
        npy_path = _with_suffix(path_stem, ".npy")
        np.save(npy_path, _as_2d(data))
        spec.data = npy_path.name
    yaml_path = _with_suffix(path_stem, ".yaml")
    _dump_yaml(spec, yaml_path)
    return yaml_path


def panel_npy_name(path_stem: str | Path, suffix: str) -> str:
    """File name ``write_figure`` gives the npy of panel ``suffix`` (for derived panels)."""
    return f"{Path(path_stem).name}_{suffix}.npy"


def write_figure(
    path_stem: str | Path, panels: list[tuple[str, np.ndarray | None, PlotSpec]], figure: FigureSpec
) -> Path:
    """Write one npy+yaml per ``(suffix, data, spec)`` panel plus ``<stem>.figure.yaml``.

    Derived panels (``data=None``) get a yaml only.
    """
    path_stem = Path(path_stem)
    figure.panels = [
        write_panel(_with_suffix(path_stem, f"_{suffix}"), data, spec).name for suffix, data, spec in panels
    ]
    figure_path = _with_suffix(path_stem, FIGURE_SUFFIX)
    _dump_yaml(figure, figure_path)
    return figure_path


def load_spec(yaml_path: str | Path) -> PlotSpec | FigureSpec:
    raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    kind = raw.get("kind") if isinstance(raw, dict) else None
    if kind == PANEL_KIND:
        return _from_dict(PlotSpec, raw)
    if kind == FIGURE_KIND:
        return _from_dict(FigureSpec, raw)
    raise ValueError(f"{yaml_path}: not a plot spec (kind={kind!r})")


def load_panel(yaml_path: str | Path) -> tuple[np.ndarray, PlotSpec]:
    yaml_path = Path(yaml_path)
    spec = load_spec(yaml_path)
    if not isinstance(spec, PlotSpec):
        raise ValueError(f"{yaml_path}: expected a panel spec, got {spec.kind!r}")
    if isinstance(spec.data, dict):
        op = spec.data.get("op")
        if op not in DERIVED_OPS:
            raise ValueError(f"{yaml_path}: unknown derived op {op!r}; choose from {sorted(DERIVED_OPS)}")
        arrays = [np.load(yaml_path.parent / name, allow_pickle=False) for name in spec.data.get("of", [])]
        return DERIVED_OPS[op](*arrays), spec
    return np.load(yaml_path.parent / spec.data, allow_pickle=False), spec
