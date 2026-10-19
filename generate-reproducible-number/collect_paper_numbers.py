#!/usr/bin/env python3
"""Collect all paper-baseline timings and step1/step3 measurements into one YAML.

Reads ``results/<variant>/<run>/`` produced by ``run_paper_baseline.py``:
  - ``timings.yaml``: wall clock per stage
  - ``timing/<stage>.yaml``: in-process timing tree written by ``python -m code``
  - ``step{1,3}/measurements.yaml``: final metrics

The top-level ``timing_overview`` lists the headline numbers per variant/run (training
time, per-epoch time, inference per sample, step2 per datapoint, ...). Plot-spec writing is
part of each stage; ``*_without_visualization_s`` subtract it.

Usage (repo root):
  .venv/bin/python generate-reproducible-number/collect_paper_numbers.py
  .venv/bin/python generate-reproducible-number/collect_paper_numbers.py -o path/to/out.yaml
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"
DEFAULT_OUT = ROOT / "paper_numbers.yaml"

RUN_ORDER = (
    "seasonal-true_vel",
    "seasonal-pred_vel",
    "steady-state-large-pred_vel",
    "steady-state-large-true_vel",
    "steady-state-small-pred_vel",
    "steady-state-small-true_vel",
)
VARIANTS = ("rwpt-high-res", "rwpt-default", "rwpt-low-res")


def _load_yaml(path: Path) -> Any | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# Headline numbers: (overview key, [(stage file, summary entry), ...] tried in order, summary field).
# Inference comes from a test-only stage if the run has one, else from the training stage's evaluation.
OVERVIEW_FIELDS = (
    ("step1_train_s", [("step1_train", "step1_train")], "training_s"),
    ("step1_epochs_run", [("step1_train", "step1_train")], "epochs_run"),
    ("step1_per_epoch_s", [("step1_train", "step1_train")], "per_epoch_s"),
    ("step1_inference_per_sample_s", [("step1_test", "step1_test"), ("step1_train", "step1_train")], "inference_test_per_sample_s"),
    ("step2_total_s", [("step2", "step2_run")], "total_s"),
    ("step2_total_without_visualization_s", [("step2", "step2_run")], "total_without_visualization_s"),
    ("step2_datapoints", [("step2", "step2_run")], "datapoints"),
    ("step2_per_datapoint_s", [("step2", "step2_run")], "per_datapoint_s"),
    ("step2_step1_inference_per_datapoint_s", [("step2", "step2_run")], "step1_inference_per_datapoint_s"),
    ("step2_rwpt_per_datapoint_s", [("step2", "step2_run")], "rwpt_per_datapoint_s"),
    ("step2_rwpt_without_visualization_per_datapoint_s", [("step2", "step2_run")], "rwpt_without_visualization_per_datapoint_s"),
    ("step3_train_s", [("step3_train", "step3_train")], "training_s"),
    ("step3_epochs_run", [("step3_train", "step3_train")], "epochs_run"),
    ("step3_per_epoch_s", [("step3_train", "step3_train")], "per_epoch_s"),
    ("step3_inference_per_sample_s", [("step3_test", "step3_test"), ("step3_train", "step3_train")], "inference_test_per_sample_s"),
)


def _stage_timing(run_dir: Path) -> dict[str, Any]:
    """``timing/<stage>.yaml`` → {stage: {device, gpu, total_s, summary}} (full tree stays on disk)."""
    stages: dict[str, Any] = {}
    timing_dir = run_dir / "timing"
    if not timing_dir.is_dir():
        return stages
    for path in sorted(timing_dir.glob("*.yaml")):
        data = _load_yaml(path) or {}
        stages[path.stem] = {
            "file": str(path.relative_to(ROOT)),
            "device": data.get("device"),
            "gpu": data.get("gpu"),
            "total_s": data.get("total_s"),
            "summary": data.get("summary", {}),
        }
    return stages


def _overview(timings: dict[str, Any] | None, stages: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, sources, field in OVERVIEW_FIELDS:
        for stage, entry in sources:
            value = stages.get(stage, {}).get("summary", {}).get(entry, {}).get(field)
            if value is not None:
                out[key] = value
                break
    if timings is not None:
        out["wall_s"] = copy.deepcopy(timings.get("wall_s", {}))
        if "step1_source" in timings:
            out["step1_source"] = timings["step1_source"]  # step1 numbers are that variant's (copied model)
    return out


def _step_measurements(run_dir: Path, step: str) -> dict[str, Any] | None:
    measurements = _load_yaml(run_dir / step / "measurements.yaml")
    if measurements is None:
        return None
    return {"measurements": measurements, "model_pt": (run_dir / step / "model.pt").is_file()}


def collect_run(variant: str, run_name: str) -> dict[str, Any]:
    run_dir = RESULTS_ROOT / variant / run_name
    entry: dict[str, Any] = {
        "available": False,
        "path": str(run_dir.relative_to(ROOT)),
    }
    if not run_dir.is_dir():
        entry["status"] = "missing"
        return entry

    timings = _load_yaml(run_dir / "timings.yaml")
    stages = _stage_timing(run_dir)
    step1 = _step_measurements(run_dir, "step1")
    step3 = _step_measurements(run_dir, "step3")
    has_step3_meas = step3 is not None
    has_timings = timings is not None

    entry["status"] = "complete" if has_step3_meas and has_timings else "incomplete"
    entry["available"] = entry["status"] == "complete"
    entry["_overview"] = _overview(timings, stages)  # collected into the top-level timing_overview
    if timings is not None:
        entry["timings"] = timings
    if stages:
        entry["stage_timing"] = stages
    if step1 is not None:
        entry["step1"] = step1
    if step3 is not None:
        entry["step3"] = step3
    return entry


def collect_all() -> dict[str, Any]:
    variants: dict[str, Any] = {}
    n_complete = 0
    n_total = 0
    for variant in VARIANTS:
        runs: dict[str, Any] = {}
        for run_name in RUN_ORDER:
            n_total += 1
            runs[run_name] = collect_run(variant, run_name)
            if runs[run_name]["available"]:
                n_complete += 1
        variants[variant] = runs

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "results_root": str(RESULTS_ROOT.relative_to(ROOT)),
        "summary": {
            "complete": n_complete,
            "total": n_total,
            "missing_or_incomplete": [
                f"{v}/{r}"
                for v, runs in variants.items()
                for r, e in runs.items()
                if not e["available"]
            ],
        },
        "timing_overview": {
            variant: {run: entry.pop("_overview") for run, entry in runs.items() if "_overview" in entry}
            for variant, runs in variants.items()
        },
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paper-baseline numbers into one YAML")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    payload = collect_all()
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    s = payload["summary"]
    print(
        f"Wrote {out}  ({s['complete']}/{s['total']} complete"
        + (f", incomplete: {s['missing_or_incomplete']}" if s["missing_or_incomplete"] else "")
        + ")",
        flush=True,
    )


if __name__ == "__main__":
    main()
