"""CLI: ``python -m code.plotting <yaml|dir>... [--format pdf] [--dpi 300]``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .render import render_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render plot spec yamls (npy + yaml) to images.")
    parser.add_argument("paths", nargs="+", type=Path, help="Spec yaml files or directories")
    parser.add_argument("--format", default=None, help="Override output format (png, pdf, svg, ...)")
    parser.add_argument("--dpi", type=float, default=None, help="Override output dpi")
    args = parser.parse_args(argv)

    for path in args.paths:
        for out in render_path(path, args.format, args.dpi):
            print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
