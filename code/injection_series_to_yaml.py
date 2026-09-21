"""Build seasonal YAML injection blocks from DaRUS CSV/NPY series.

Reads ``temperature_injection_series`` (ΔT vs ambient [°C]) and
``normed_flow_injection_series`` (unitless; scaled by ``NORMED_FLOW_SCALE``).

Series layout (DaRUS full / step-1 injection schedules)
------------------------------------------------------
- Length 731 = **10 years × 73 samples/year + 1 wrap-around** duplicate of the
  first sample (last == first; ignore for duration counting).
- Do **not** use ``dataset_info`` ``timeresolution [d]: 18.25`` here — that is the
  *field snapshot* spacing, not the injection-series sampling.

Extracts one calendar year (73 steps + endpoint at t=1.0 y) for
``injection_temperature_C`` / ``injection_rate_m3_per_s`` in ``settings/*.yaml``.

Example
-------
python code/injection_series_to_yaml.py \\
  --temp datasets/full_dataset/general/temperature_injection_series.csv \\
  --flow datasets/full_dataset/general/normed_flow_injection_series.csv \\
  --year 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_AMBIENT_C = 10.0
# DaRUS normed flow (~1–7) -> physical injection rate [m³/s].
NORMED_FLOW_SCALE = 0.0005
# Injection series: 73 samples per calendar year (not 20 from 18.25 d field dt).
DEFAULT_STEPS_PER_YEAR = 73


def load_series(path: Path) -> np.ndarray:
    """Load a 1-D injection series from ``.npy`` or text/CSV."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        data = np.load(path, allow_pickle=False)
    else:
        data = np.loadtxt(path)
    if data.ndim != 1:
        raise ValueError(f"{path}: expected a 1-D series, got shape {data.shape}")
    return data.astype(np.float64)


def steps_per_year_from_series(series: np.ndarray, *, default: int = DEFAULT_STEPS_PER_YEAR) -> int:
    """Infer samples/year from length (prefer 73 when (len-1) divides evenly)."""
    n = len(series)
    if n >= 2 and np.isclose(series[0], series[-1]):
        body = n - 1
    else:
        body = n
    if body % default == 0 and body // default >= 1:
        return default
    # Legacy fallback: 18.25 d field-style cadence (~20 / year) if it divides.
    for candidate in (20, default):
        if body % candidate == 0 and body // candidate >= 1:
            return candidate
    raise ValueError(
        f"Cannot infer steps/year from series length {n} "
        f"(body={body}); expected k*{default}+optional wrap"
    )


def pick_year_strongest_swing(temp_delta: np.ndarray, n_steps: int) -> int:
    """Choose the calendar year with the largest ΔT range (warm + cold)."""
    body = (
        len(temp_delta) - 1
        if (len(temp_delta) >= 2 and np.isclose(temp_delta[0], temp_delta[-1]))
        else len(temp_delta)
    )
    n_years = body // n_steps
    if n_years < 1:
        raise ValueError("Series too short for one full year")
    best_year, best_span = 0, -1.0
    for year in range(n_years):
        seg = temp_delta[year * n_steps : (year + 1) * n_steps]
        span = float(seg.max() - seg.min())
        if span > best_span:
            best_year, best_span = year, span
    return best_year


def extract_year_cycle(
    temp_delta: np.ndarray,
    flow_norm: np.ndarray,
    *,
    year: int,
    n_steps: int,
    ambient_c: float,
    flow_scale: float = NORMED_FLOW_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times_year, temp_C_abs, rate_m3_per_s) with length n_steps+1.

    The extra sample at t=1.0 is the start of the next year (or the wrap-around
    duplicate equal to series[0] when ``year`` is the last year).
    """
    if len(temp_delta) != len(flow_norm):
        raise ValueError(
            f"Series length mismatch: temp={len(temp_delta)} flow={len(flow_norm)}"
        )
    start = year * n_steps
    end = start + n_steps + 1  # include start of next year / wrap as t=1.0
    if end > len(temp_delta):
        raise ValueError(
            f"year={year} needs indices [{start}:{end}) but series length is {len(temp_delta)}"
        )

    times = np.linspace(0.0, 1.0, n_steps + 1)
    temp_abs = ambient_c + temp_delta[start:end]
    rate_m3_s = flow_norm[start:end] * flow_scale
    return times, temp_abs, rate_m3_s


def format_values_block(times: np.ndarray, values: np.ndarray, indent: int = 10) -> str:
    sp = " " * indent
    return "\n".join(f"{sp}{t:.5f}: {v:.10g}" for t, v in zip(times, values, strict=True))


def format_yaml_fragment(
    times: np.ndarray,
    temp_abs: np.ndarray,
    rate_m3_s: np.ndarray,
    *,
    indent: int = 6,
) -> str:
    """YAML under ``physical_parameters`` (default indent matches seasonal-*.yaml)."""
    sp = " " * indent
    sp2 = " " * (indent + 2)
    temp_block = format_values_block(times, temp_abs, indent=indent + 4)
    rate_block = format_values_block(times, rate_m3_s, indent=indent + 4)
    return (
        f"{sp}injection_temperature_C:\n"
        f"{sp2}time_unit: year\n"
        f"{sp2}values:\n"
        f"{temp_block}\n"
        f"{sp}injection_rate_m3_per_s:\n"
        f"{sp2}time_unit: year\n"
        f"{sp2}values:\n"
        f"{rate_block}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temp",
        type=Path,
        default=Path("datasets/full_dataset/general/temperature_injection_series.csv"),
        help="Path to temperature_injection_series (ΔT [°C], .csv or .npy)",
    )
    parser.add_argument(
        "--flow",
        type=Path,
        default=Path("datasets/full_dataset/general/normed_flow_injection_series.csv"),
        help="Path to normed_flow_injection_series (unitless, .csv or .npy)",
    )
    parser.add_argument(
        "--ambient",
        type=float,
        default=DEFAULT_AMBIENT_C,
        help=f"Ambient temperature [°C] added to ΔT (default {DEFAULT_AMBIENT_C})",
    )
    parser.add_argument(
        "--steps-per-year",
        type=int,
        default=None,
        help=f"Samples per year (default: infer, usually {DEFAULT_STEPS_PER_YEAR})",
    )
    parser.add_argument(
        "--flow-scale",
        type=float,
        default=NORMED_FLOW_SCALE,
        help=f"Multiply normed flow by this to get m³/s (default {NORMED_FLOW_SCALE})",
    )
    parser.add_argument(
        "--year",
        default="auto",
        help="0-based calendar year index, or 'auto' for strongest ΔT swing",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=6,
        help="Spaces before injection_* keys (6 matches seasonal-*.yaml physical_parameters)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write YAML fragment here (default: stdout)",
    )
    args = parser.parse_args()

    temp_delta = load_series(args.temp)
    flow_norm = load_series(args.flow)
    n_steps = args.steps_per_year or steps_per_year_from_series(temp_delta)

    if str(args.year).lower() == "auto":
        year = pick_year_strongest_swing(temp_delta, n_steps)
    else:
        year = int(args.year)

    times, temp_abs, rate = extract_year_cycle(
        temp_delta,
        flow_norm,
        year=year,
        n_steps=n_steps,
        ambient_c=args.ambient,
        flow_scale=args.flow_scale,
    )
    fragment = format_yaml_fragment(times, temp_abs, rate, indent=args.indent)

    header = (
        f"# From {args.temp.name} + {args.flow.name}\n"
        f"# year={year}, ambient={args.ambient} C, steps/year={n_steps}, "
        f"flow_scale={args.flow_scale}\n"
        f"# ΔT series range [{temp_delta.min():.4g}, {temp_delta.max():.4g}] C → "
        f"abs temp [{temp_abs.min():.4g}, {temp_abs.max():.4g}] C, "
        f"rate [{rate.min():.4g}, {rate.max():.4g}] m^3/s\n"
    )
    text = header + fragment

    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output} (year={year})")


if __name__ == "__main__":
    main()
