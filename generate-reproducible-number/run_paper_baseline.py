#!/usr/bin/env python3
"""Run the pLGCNN paper cases at three RWPT budgets (high-res, default, low-res).

Configs: ``config/<variant>/<run>.yaml``; results: ``results/<variant>/<run>/``.

Per (variant, run):
  - no result folder: train step1 (pred_vel only) → step2 → step3; with ``--reuse-step1 <variant>``
    pred_vel runs copy that variant's step1 model instead of training it (step1 does not depend
    on the RWPT settings)
  - folder without step3/measurements.yaml or timings.yaml: re-evaluate the existing models
  - otherwise: skip

After every stage and run a short notification with the estimated remaining time is printed and,
if ``<repo>/.pushover`` holds a Pushover URL, sent (``baseline_progress.py``).

Steps with ``visualize: false`` are rendered afterwards by ``python -m code.plotting`` (untimed,
``render_s`` in ``timings.yaml``). Each call's ``timing.yaml`` is kept as ``timing/<stage>.yaml``;
``timings.yaml`` holds the wall clock per stage.

Usage (repo root):
  .venv/bin/python generate-reproducible-number/run_paper_baseline.py [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
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
from baseline_progress import Notifier, Progress

# One job per GPU at a time.
GPUS = [1, 2, 3]

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
VARIANTS = ("rwpt-high-res", "rwpt-default", "rwpt-low-res")

# tqdm bars in the ``python -m code`` calls (``--progress-bars``).
PROGRESS_BARS = False
TIMINGS_FILE = "timings.yaml"
# Copied by --reuse-step1: what step2 needs (model + normalization) plus the metrics.
STEP1_REUSE_FILES = ("model.pt", "info.yaml", "measurements.yaml", "model_structure.txt")
# "<variant>/<run>" of failed jobs (shared by the GPU workers).
FAILED_RUNS: set[str] = set()

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def git_state() -> dict[str, Any]:
    """Commit of the code that produced the numbers; ``dirty`` = uncommitted changes to tracked files."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _load_template(variant: str, run_name: str) -> dict:
    """Load ``config/<variant>/<run>.yaml`` with includes resolved."""
    from code.utils.yaml_includes import load_yaml_with_includes

    path = CONFIG_ROOT / variant / f"{run_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return copy.deepcopy(load_yaml_with_includes(path))


def _write_working_cfg(run_dir: Path, cfg: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "working_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _renders_images(cfg: dict, step: str) -> bool:
    """``visualize`` of ``step``: images are rendered during the run."""
    return bool(cfg.get("general_configuration", {}).get(step, {}).get("general", {}).get("visualize", False))


def _prepare_test_metrics(cfg: dict, step: str) -> None:
    """Evaluate on full fields with batch size 1."""
    mp = cfg["general_configuration"][step]["model_parameters"]
    mp["batchsize"] = 1
    mp["bool_cutouts"] = False


def render_specs(step_dir: Path) -> None:
    """Render all plot specs (npy + yaml) below ``step_dir`` to images."""
    subprocess.run([str(PYTHON), "-m", "code.plotting", str(step_dir)], cwd=str(REPO), check=True)


def run_code(cfg_path: Path) -> None:
    env = os.environ.copy()
    if not PROGRESS_BARS:
        env.setdefault("TQDM_DISABLE", "1")  # tqdm treats any non-empty value as "disable"
    subprocess.run([str(PYTHON), "-m", "code", str(cfg_path)], cwd=str(REPO), check=True, env=env)


class JobRunner:
    def __init__(
        self,
        variant: str,
        run_name: str,
        gpu: int,
        dry_run: bool = False,
        *,
        from_step2: bool = False,
        step1_source: str | None = None,
        progress: Progress | None = None,
    ):
        self.variant = variant
        self.run_name = run_name
        self.gpu = gpu
        self.device = f"cuda:{gpu}"
        self.dry_run = dry_run
        self.from_step2 = from_step2
        self.step1_source = step1_source if step1_source != variant else None
        self.run_dir = RESULTS_ROOT / variant / run_name
        self.timings: dict[str, int] = {}
        self.render_timings: dict[str, int] = {}
        self.unrendered_steps: set[str] = set()  # steps that ran with visualize: false
        self.mode = "unknown"
        self.progress = progress

    def planned_stages(self) -> list[str]:
        """Timed stages ``run`` will execute (same decision as ``run``)."""
        pred = self.use_velocity_model()
        if self.from_step2 or (not self.run_dir.exists() and (self.step1_source or not pred)):
            return ["step2", "step3_train"]
        if not self.run_dir.exists():
            return ["step1_train", "step2", "step3_train"]
        if self.is_complete():
            return []
        return (["step1_test"] if pred else []) + ["step3_test"]

    def _invoke(self, cfg: dict, stage: str) -> None:
        cfg = copy.deepcopy(cfg)
        cfg["run_configuration"]["device"] = self.device
        if self.dry_run:
            print(
                f"[dry-run] {self.variant}/{self.run_name} {stage} "
                f"pipeline={cfg['run_configuration']['pipeline']} device={self.device}",
                flush=True,
            )
            return
        path = _write_working_cfg(self.run_dir, cfg)
        run_code(path)
        self._keep_timing(stage)

    def _keep_timing(self, stage: str) -> None:
        """Move this call's ``timing.yaml`` to ``timing/<stage>.yaml``."""
        src = self.run_dir / "timing.yaml"
        if not src.is_file():
            print(f"WARN  no timing.yaml after {self.variant}/{self.run_name} {stage}", flush=True)
            return
        dst_dir = self.run_dir / "timing"
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dst_dir / f"{stage}.yaml")

    def _timed(self, stage: str, cfg: dict) -> None:
        t0 = time.time()
        if self.progress is not None and not self.dry_run:
            self.progress.stage_started(self.variant, self.run_name, stage)
        self._invoke(cfg, stage)
        self.timings[stage] = int(round(time.time() - t0))
        if self.progress is not None and not self.dry_run:
            self.progress.stage_done(self.variant, self.run_name, stage, self.timings[stage])

    def render_step(self, step: str) -> None:
        """Render ``<run>/<step>`` (untimed) if that step ran with ``visualize: false``; failures only warn."""
        if step not in self.unrendered_steps:
            return
        self.unrendered_steps.discard(step)
        step_dir = self.run_dir / step
        if self.dry_run:
            print(f"[dry-run] {self.variant}/{self.run_name} render {step_dir}", flush=True)
            return
        t0 = time.time()
        try:
            render_specs(step_dir)
        except subprocess.CalledProcessError as exc:
            print(f"WARN  rendering {step_dir} failed: {exc}", flush=True)
        self.render_timings[step] = int(round(time.time() - t0))

    def _cfg_with_pipeline(self, pipeline: list[dict]) -> dict:
        cfg = _load_template(self.variant, self.run_name)
        cfg["run_configuration"]["pipeline"] = pipeline
        cfg["run_configuration"]["device"] = self.device
        for action in pipeline:
            for step in action:
                if step in ("step1", "step2", "step3") and not _renders_images(cfg, step):
                    self.unrendered_steps.add(step)
        return cfg

    def use_velocity_model(self) -> bool:
        cfg = _load_template(self.variant, self.run_name)
        return bool(cfg["run_configuration"].get("use_velocity_model"))

    def is_complete(self) -> bool:
        return (self.run_dir / "step3" / "measurements.yaml").is_file() and (self.run_dir / TIMINGS_FILE).is_file()

    def write_timings(self) -> None:
        if self.dry_run:
            return
        timing_dir = self.run_dir / "timing"
        payload = {
            "run": self.run_name,
            "variant": self.variant,
            "device": self.device,
            "mode": self.mode,
            "finished_at": now(),
            "git": git_state(),
            "wall_s": self.timings,
            **({"step1_source": self.step1_source} if self.mode == "reuse_step1" else {}),
            "render_s": self.render_timings,  # untimed image rendering, not part of wall_s stages
            "timing_files": sorted(p.name for p in timing_dir.glob("*.yaml")) if timing_dir.is_dir() else [],
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / TIMINGS_FILE).write_text(yaml.safe_dump(payload, sort_keys=False))

    def _test_step(self, step: str, *, timed_name: str) -> None:
        cfg = self._cfg_with_pipeline([{step: "test"}])
        _prepare_test_metrics(cfg, step)
        self._timed(timed_name, cfg)

    def _run_step2(self) -> None:
        self._timed("step2", self._cfg_with_pipeline([{"clean": "results-step2"}, {"step2": "run"}]))

    def run_full(self) -> None:
        if not self.dry_run:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "data_prep").mkdir(parents=True, exist_ok=True)
        use_vel = self.use_velocity_model()

        if use_vel:
            self._timed(
                "step1_train",
                self._cfg_with_pipeline([{"step1": "train"}]),
            )
            self.render_step("step1")

        self._run_step2()
        self.render_step("step2")
        self._timed(
            "step3_train",
            self._cfg_with_pipeline([{"clean": "results-step3"}, {"step3": "train"}]),
        )
        self.render_step("step3")

    def _copy_step1(self) -> None:
        """Copy the step1 model (and its training timing) of the same run in ``step1_source``."""
        src = RESULTS_ROOT / self.step1_source / self.run_name
        model = src / "step1" / "model.pt"
        if self.dry_run:
            print(f"[dry-run] {self.variant}/{self.run_name} copy step1 from {src}", flush=True)
            return
        # On several GPUs the source run may still be training: wait while it is in progress.
        while (
            not model.is_file()
            and src.is_dir()
            and not (src / TIMINGS_FILE).is_file()
            and f"{self.step1_source}/{self.run_name}" not in FAILED_RUNS
        ):
            time.sleep(60)
        if not model.is_file():
            raise FileNotFoundError(f"Missing {model}; run --variants {self.step1_source} first")
        (self.run_dir / "step1").mkdir(parents=True, exist_ok=True)
        for name in STEP1_REUSE_FILES:
            if (src / "step1" / name).is_file():
                shutil.copy2(src / "step1" / name, self.run_dir / "step1" / name)
        if (src / "timing" / "step1_train.yaml").is_file():
            (self.run_dir / "timing").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "timing" / "step1_train.yaml", self.run_dir / "timing" / "step1_train.yaml")

    def run_reuse_step1(self) -> None:
        self._copy_step1()
        if not self.dry_run:
            (self.run_dir / "data_prep").mkdir(parents=True, exist_ok=True)
        self._run_step2()
        self.render_step("step2")
        self._timed(
            "step3_train",
            self._cfg_with_pipeline([{"clean": "results-step3"}, {"step3": "train"}]),
        )
        self.render_step("step3")

    def run_test_only(self) -> None:
        use_vel = self.use_velocity_model()
        if use_vel:
            if not (self.run_dir / "step1" / "model.pt").is_file():
                raise FileNotFoundError(f"Missing {self.run_dir / 'step1' / 'model.pt'} for test-only")
            self._test_step("step1", timed_name="step1_test")
            self.render_step("step1")
        if not (self.run_dir / "step3" / "model.pt").is_file():
            raise FileNotFoundError(f"Missing {self.run_dir / 'step3' / 'model.pt'} for test-only")
        self._test_step("step3", timed_name="step3_test")
        self.render_step("step3")

    def run_from_step2(self) -> None:
        """Reuse existing step1 (if any); re-run step2 → step3 train."""
        if not self.dry_run:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "data_prep").mkdir(parents=True, exist_ok=True)
        if self.use_velocity_model() and not (self.run_dir / "step1" / "model.pt").is_file():
            raise FileNotFoundError(
                f"Missing {self.run_dir / 'step1' / 'model.pt'} for --from-step2 (pred_vel needs step1)"
            )
        self._run_step2()
        self.render_step("step2")
        self._timed(
            "step3_train",
            self._cfg_with_pipeline([{"clean": "results-step3"}, {"step3": "train"}]),
        )
        self.render_step("step3")

    def run(self) -> str:
        label = f"{self.variant}/{self.run_name}"
        if self.from_step2:
            self.mode, action, note = "from_step2", self.run_from_step2, ""
        elif not self.run_dir.exists() and self.step1_source and self.use_velocity_model():
            self.mode, action, note = "reuse_step1", self.run_reuse_step1, f" (step1 from {self.step1_source})"
        elif not self.run_dir.exists():
            self.mode, action, note = "full", self.run_full, ""
        elif self.is_complete():
            print(f"SKIP  {label}", flush=True)
            return f"skip:{label}"
        else:
            self.mode, action, note = "test_only", self.run_test_only, ""
        print(f"{self.mode.upper()}  {label} on {self.device}{note}", flush=True)
        t0 = time.time()
        action()
        self.timings["total"] = int(round(time.time() - t0))
        self.write_timings()
        return f"{self.mode}:{label}"


def _worker(
    gpu: int,
    jobs: queue.Queue,
    dry_run: bool,
    results: list,
    lock: threading.Lock,
    *,
    from_step2: bool = False,
    step1_source: str | None = None,
    progress: Progress | None = None,
) -> None:
    while True:
        try:
            variant, run_name = jobs.get_nowait()
        except queue.Empty:
            return
        t0 = time.time()
        try:
            status = JobRunner(
                variant, run_name, gpu, dry_run=dry_run, from_step2=from_step2, step1_source=step1_source,
                progress=progress,
            ).run()
            with lock:
                results.append({"status": status, "gpu": gpu, "ok": True})
            if progress is not None and not dry_run:
                progress.job_done(variant, run_name, status, time.time() - t0)
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
            FAILED_RUNS.add(f"{variant}/{run_name}")
            print(f"FAIL  {variant}/{run_name} on cuda:{gpu}: {exc}", flush=True)
            if progress is not None and not dry_run:
                progress.job_done(variant, run_name, "fail", time.time() - t0, error=str(exc))
        finally:
            jobs.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper baseline")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions only")
    parser.add_argument(
        "--from-step2",
        action="store_true",
        help="Skip step1; re-run step2 → step3 train (reuse existing step1 models)",
    )
    parser.add_argument(
        "--reuse-step1",
        type=str,
        default="",
        metavar="VARIANT",
        help="New pred_vel runs copy step1 from this variant instead of training it (e.g. rwpt-default)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated run names to include (default: all)",
    )
    parser.add_argument(
        "--progress-bars",
        action="store_true",
        help="Show tqdm progress bars (default: off, per-epoch log lines only)",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Do not send Pushover notifications (they are still printed)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help=f"Comma-separated GPU indices, one job per GPU at a time (default: {GPUS})",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="Comma-separated variants (default: all VARIANTS)",
    )
    args = parser.parse_args()
    global PROGRESS_BARS
    PROGRESS_BARS = args.progress_bars

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    variants = [s.strip() for s in args.variants.split(",") if s.strip()] or list(VARIANTS)
    runs = [r for r in RUN_ORDER if not only or r in only]
    gpus = [int(s) for s in args.gpus.split(",") if s.strip()] or list(GPUS)
    step1_source = args.reuse_step1.strip() or None
    if step1_source is not None:
        if step1_source not in VARIANTS:
            parser.error(f"--reuse-step1: unknown variant {step1_source!r}")
        # the source variant trains its step1 models first
        variants.sort(key=lambda v: v != step1_source)

    job_q: queue.Queue = queue.Queue()
    plans: dict[tuple[str, str], list[str]] = {}
    for variant in variants:
        for run_name in runs:
            job_q.put((variant, run_name))
            plans[(variant, run_name)] = JobRunner(
                variant, run_name, gpus[0], from_step2=args.from_step2, step1_source=step1_source
            ).planned_stages()
    notifier = Notifier(REPO / ".pushover", enabled=not (args.dry_run or args.no_notify))
    progress = Progress(plans, len(gpus), notifier)

    git = git_state()
    print(
        f"paper baseline  commit={git['commit'][:10]}{' (DIRTY)' if git['dirty'] else ''}  "
        f"GPUS={gpus}  jobs={job_q.qsize()}  from_step2={args.from_step2}  "
        f"reuse_step1={step1_source}  "
        f"results={RESULTS_ROOT}  notify={'on' if notifier.url else 'off'}",
        flush=True,
    )
    if not args.dry_run:
        progress.started()

    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    t0 = time.time()
    threads = [
        threading.Thread(
            target=_worker,
            args=(gpu, job_q, args.dry_run, results, lock),
            kwargs={"from_step2": args.from_step2, "step1_source": step1_source, "progress": progress},
            daemon=True,
        )
        for gpu in gpus
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    total = int(round(time.time() - t0))
    summary = {
        "finished_at": now(),
        "git": git_state(),
        "gpus": gpus,
        "total_wall_s": total,
        "jobs": results,
    }
    n_fail = sum(1 for r in results if not r.get("ok", True))
    if args.dry_run:
        print(f"Done (dry-run). failures={n_fail}", flush=True)
    else:
        summary_path = RESULTS_ROOT / "pipeline_timings.yaml"
        summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
        print(f"Done ({total}s). summary={summary_path} failures={n_fail}", flush=True)
        progress.all_done(total, n_fail)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
