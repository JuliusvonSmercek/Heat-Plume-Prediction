#!/usr/bin/env python3
"""Produce paper-comparable pLGCNN numbers (RWPT high-res + low-res).

Layout:
  generate-reproducible-number/
    config/rwpt-high-res|rwpt-low-res/<run>.{train,test}.yaml
    results/<variant>/<run>/{step1,step2,step3,data_prep,timings.json}

Per (variant, run):
  - missing run folder          → full from scratch
  - folder exists, incomplete   → test only (no-viz timed, then viz untimed)
  - measurements.yaml + timings.json → skip

Usage (from repo root, in tmux):
  .venv/bin/python generate-reproducible-number/run_paper_baseline.py
  .venv/bin/python generate-reproducible-number/run_paper_baseline.py --dry-run
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# One job per GPU at a time.
GPUS = [3]

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CONFIG_ROOT = ROOT / "config"
RESULTS_ROOT = ROOT / "results"
LOG_DIR = ROOT / "logs"
PYTHON = REPO / ".venv" / "bin" / "python"
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)

RUN_ORDER = (
    "seasonal-true_vel",
    "seasonal-pred_vel",
    "steady-state-large-pred_vel",
    "steady-state-large-true_vel",
    "steady-state-small-pred_vel",
    "steady-state-small-true_vel",
)
VARIANTS = ("rwpt-high-res", "rwpt-low-res")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_template(variant: str, run_name: str, kind: str) -> dict:
    from code.utils.yaml_includes import load_yaml_with_includes

    path = CONFIG_ROOT / variant / f"{run_name}.{kind}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return copy.deepcopy(load_yaml_with_includes(path))


def _write_working_cfg(run_dir: Path, cfg: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "working_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _set_visualize(cfg: dict, *, visualize: bool) -> None:
    for step in ("step1", "step3"):
        if step in cfg.get("general_configuration", {}):
            gen = cfg["general_configuration"][step].setdefault("general", {})
            gen["visualize"] = visualize
            gen["visualize_epochs"] = False


def _prepare_test_metrics(cfg: dict, step: str) -> None:
    mp = cfg["general_configuration"][step]["model_parameters"]
    mp["batchsize"] = 1
    mp["bool_cutouts"] = False


def run_code(cfg_path: Path, *, env: dict[str, str] | None = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(
        [str(PYTHON), "-m", "code", str(cfg_path)],
        cwd=str(REPO),
        check=True,
        env=merged,
    )


class JobRunner:
    def __init__(self, variant: str, run_name: str, gpu: int, dry_run: bool = False):
        self.variant = variant
        self.run_name = run_name
        self.gpu = gpu
        self.device = f"cuda:{gpu}"
        self.dry_run = dry_run
        self.run_dir = RESULTS_ROOT / variant / run_name
        self.timings: dict[str, int] = {}
        self.mode = "unknown"

    def _invoke(self, cfg: dict) -> None:
        cfg = copy.deepcopy(cfg)
        cfg["run_configuration"]["device"] = self.device
        if self.dry_run:
            print(
                f"[dry-run] {self.variant}/{self.run_name} "
                f"pipeline={cfg['run_configuration']['pipeline']} device={self.device}",
                flush=True,
            )
            return
        path = _write_working_cfg(self.run_dir, cfg)
        run_code(path)

    def _timed(self, stage: str, cfg: dict) -> None:
        t0 = time.time()
        self._invoke(cfg)
        self.timings[stage] = int(round(time.time() - t0))

    def _cfg_with_pipeline(self, kind: str, pipeline: list[dict], *, visualize: bool = False) -> dict:
        cfg = _load_template(self.variant, self.run_name, kind)
        cfg["run_configuration"]["pipeline"] = pipeline
        cfg["run_configuration"]["device"] = self.device
        _set_visualize(cfg, visualize=visualize)
        return cfg

    def use_velocity_model(self) -> bool:
        cfg = _load_template(self.variant, self.run_name, "train")
        return bool(cfg["run_configuration"].get("use_velocity_model"))

    def is_complete(self) -> bool:
        return (self.run_dir / "step3" / "measurements.yaml").is_file() and (
            self.run_dir / "timings.json"
        ).is_file()

    def write_timings(self) -> None:
        if self.dry_run:
            return
        payload = {
            "run": self.run_name,
            "variant": self.variant,
            "device": self.device,
            "mode": self.mode,
            "finished_at": now(),
            "wall_s": self.timings,
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "timings.json").write_text(json.dumps(payload, indent=2) + "\n")

    def _copy_measurements_train(self, step: str) -> None:
        if self.dry_run:
            return
        src = self.run_dir / step / "measurements.yaml"
        if src.is_file():
            shutil.copy2(src, self.run_dir / step / "measurements_train.yaml")

    def _test_step(self, step: str, *, timed_name: str) -> None:
        # no-viz timed
        cfg = self._cfg_with_pipeline("test" if step == "step3" else "train", [{step: "test"}], visualize=False)
        _prepare_test_metrics(cfg, step)
        self._timed(timed_name, cfg)
        # viz untimed
        cfg_v = self._cfg_with_pipeline("test" if step == "step3" else "train", [{step: "test"}], visualize=True)
        _prepare_test_metrics(cfg_v, step)
        self._invoke(cfg_v)

    def run_full(self) -> None:
        self.mode = "full"
        if not self.dry_run:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "data_prep").mkdir(parents=True, exist_ok=True)
        use_vel = self.use_velocity_model()

        if use_vel:
            self._timed(
                "step1_train",
                self._cfg_with_pipeline("train", [{"step1": "train"}], visualize=False),
            )
            self._copy_measurements_train("step1")
            self._test_step("step1", timed_name="step1_test")

        self._timed(
            "step2",
            self._cfg_with_pipeline(
                "train",
                [{"clean": "results-step2"}, {"step2": "run"}],
                visualize=False,
            ),
        )
        self._timed(
            "step3_train",
            self._cfg_with_pipeline(
                "train",
                [{"clean": "results-step3"}, {"step3": "train"}],
                visualize=False,
            ),
        )
        self._copy_measurements_train("step3")
        self._test_step("step3", timed_name="step3_test")
        self.write_timings()

    def run_test_only(self) -> None:
        self.mode = "test_only"
        use_vel = self.use_velocity_model()
        if use_vel:
            if not (self.run_dir / "step1" / "model.pt").is_file():
                raise FileNotFoundError(f"Missing {self.run_dir / 'step1' / 'model.pt'} for test-only")
            self._test_step("step1", timed_name="step1_test")
        if not (self.run_dir / "step3" / "model.pt").is_file():
            raise FileNotFoundError(f"Missing {self.run_dir / 'step3' / 'model.pt'} for test-only")
        self._test_step("step3", timed_name="step3_test")
        self.write_timings()

    def run(self) -> str:
        label = f"{self.variant}/{self.run_name}"
        if not self.run_dir.exists():
            print(f"FULL  {label} on {self.device}", flush=True)
            t0 = time.time()
            self.run_full()
            self.timings["total"] = int(round(time.time() - t0))
            self.write_timings()
            return f"full:{label}"
        if self.is_complete():
            print(f"SKIP  {label}", flush=True)
            return f"skip:{label}"
        print(f"TEST  {label} on {self.device}", flush=True)
        t0 = time.time()
        self.run_test_only()
        self.timings["total"] = int(round(time.time() - t0))
        self.write_timings()
        return f"test:{label}"


def _worker(gpu: int, jobs: queue.Queue, dry_run: bool, results: list, lock: threading.Lock) -> None:
    while True:
        try:
            variant, run_name = jobs.get_nowait()
        except queue.Empty:
            return
        try:
            status = JobRunner(variant, run_name, gpu, dry_run=dry_run).run()
            with lock:
                results.append({"status": status, "gpu": gpu, "ok": True})
        except Exception as exc:  # noqa: BLE001 — record and continue other jobs
            with lock:
                results.append(
                    {
                        "status": f"fail:{variant}/{run_name}",
                        "gpu": gpu,
                        "ok": False,
                        "error": str(exc),
                    }
                )
            print(f"FAIL  {variant}/{run_name} on cuda:{gpu}: {exc}", flush=True)
        finally:
            jobs.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper baseline RWPT high/low orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions only")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated run names to include (default: all)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="Comma-separated variants (default: rwpt-high-res,rwpt-low-res)",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    variants = [s.strip() for s in args.variants.split(",") if s.strip()] or list(VARIANTS)
    runs = [r for r in RUN_ORDER if not only or r in only]

    job_q: queue.Queue = queue.Queue()
    for variant in variants:
        for run_name in runs:
            job_q.put((variant, run_name))

    print(
        f"paper baseline  GPUS={GPUS}  jobs={job_q.qsize()}  results={RESULTS_ROOT}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    t0 = time.time()
    threads = [
        threading.Thread(target=_worker, args=(gpu, job_q, args.dry_run, results, lock), daemon=True)
        for gpu in GPUS
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    total = int(round(time.time() - t0))
    summary = {
        "finished_at": now(),
        "gpus": GPUS,
        "total_wall_s": total,
        "jobs": results,
    }
    summary_path = RESULTS_ROOT / "pipeline_timings.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    n_fail = sum(1 for r in results if not r.get("ok", True))
    print(f"Done ({total}s). summary={summary_path} failures={n_fail}", flush=True)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
