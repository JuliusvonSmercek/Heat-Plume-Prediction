"""Standalone CLI to difference and visualize two physical-temperature ``.npy`` fields.

Also used offline against step2 outputs:

    .venv/bin/python code/compare_temperature_fields.py \\
      results/.../RUN_0/thermal_prior_C.npy \\
      results/.../RUN_0/temperature_true_C.npy \\
      -o results/.../RUN_0/temperature_diff.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from code.streamlines.environment_pre import compare_temperature_fields


def load_field(path: Path) -> np.ndarray:
    """Load a 2-D float field from ``.npy``."""
    path = Path(path)
    arr = np.squeeze(np.asarray(np.load(path, allow_pickle=False), dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D field after squeeze, got shape {arr.shape}")
    return arr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Absolute-difference visualization of two temperature field .npy files."
    )
    parser.add_argument("field_a", type=Path, help="First field (.npy), e.g. thermal_prior_C.npy")
    parser.add_argument("field_b", type=Path, help="Second field (.npy), e.g. temperature_true_C.npy")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <stem_a>_vs_<stem_b>.png next to field_a)",
    )
    parser.add_argument("--title-a", default=None, help="Panel title for field A")
    parser.add_argument("--title-b", default=None, help="Panel title for field B")
    args = parser.parse_args(argv)

    a = load_field(args.field_a)
    b = load_field(args.field_b)
    title_a = args.title_a or args.field_a.stem
    title_b = args.title_b or args.field_b.stem
    out_path = args.output
    if out_path is None:
        out_path = args.field_a.parent / f"{args.field_a.stem}_vs_{args.field_b.stem}.png"

    metrics = compare_temperature_fields(
        a, b, title_a=title_a, title_b=title_b, out_path=out_path
    )
    print(f"shape: {metrics['shape']}")
    print(f"MAE:     {metrics['mae']:.6g}")
    print(f"RMSE:    {metrics['rmse']:.6g}")
    print(f"max|Δ|:  {metrics['max_abs']:.6g}")
    print(f"wrote:   {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
