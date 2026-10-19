"""Self-contained plotting: plot specs (``<stem>.npy`` + ``<stem>.yaml``) and their renderer.

Depends only on numpy, yaml, matplotlib and scipy (no other ``code.*`` imports), so the
folder can be copied elsewhere to re-render figures. Edit a spec yaml and re-render with
``python -m code.plotting <yaml|dir> [--format pdf] [--dpi 300]``.
"""

from .spec import (
    AxesSpec,
    ColormapSpec,
    CutoutSpec,
    DisplaySpec,
    FigureSpec,
    OutputSpec,
    OverlaySpec,
    PlotSpec,
    load_panel,
    load_spec,
    write_figure,
    write_panel,
)

__all__ = [
    "AxesSpec",
    "ColormapSpec",
    "CutoutSpec",
    "DisplaySpec",
    "FigureSpec",
    "OutputSpec",
    "OverlaySpec",
    "PlotSpec",
    "load_panel",
    "load_spec",
    "write_figure",
    "write_panel",
]
