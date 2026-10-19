"""Progress tracking, remaining-time estimate and Pushover notifications for ``run_paper_baseline.py``.

Notifications go to the URL in ``<repo>/.pushover`` (one line, gitignored), e.g.
``https://api.pushover.net/1/messages.json?token=<app token>&user=<user key>``. The query
parameters are sent as form fields together with ``title`` and ``message``. Without a valid
URL nothing is sent; a failed request only prints a warning.
"""
from __future__ import annotations

import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

TITLE = "pLGCNN baseline"

# Rough stage durations [s] on one GPU, used until a stage of the same kind has been measured
# in this invocation. step1/step3 per dataset, step2 per (variant, dataset).
DEFAULT_TRAIN_S = {
    ("step1_train", "seasonal"): 36_000,
    ("step1_train", "steady-state-large"): 10_000,
    ("step1_train", "steady-state-small"): 36_000,
    ("step3_train", "seasonal"): 23_000,
    ("step3_train", "steady-state-large"): 13_000,
    ("step3_train", "steady-state-small"): 23_000,
}
DEFAULT_STEP2_S = {
    "rwpt-high-res": {"seasonal": 1_800, "steady-state-large": 800, "steady-state-small": 1_800},
    "rwpt-default": {"seasonal": 270, "steady-state-large": 140, "steady-state-small": 270},
    "rwpt-low-res": {"seasonal": 180, "steady-state-large": 40, "steady-state-small": 180},
}
DEFAULT_TEST_S = 120
RENDER_S = 120  # untimed rendering after a step, per stage


def dataset_of(run_name: str) -> str:
    return run_name.rsplit("-", 1)[0]


def format_duration(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    days, rest = divmod(minutes, 24 * 60)
    hours, minutes = divmod(rest, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


class Notifier:
    def __init__(self, config_path: Path, enabled: bool = True):
        self.url = None
        if enabled and config_path.is_file():
            lines = [ln.strip() for ln in config_path.read_text().splitlines()]
            url = next((ln for ln in lines if ln and not ln.startswith("#")), "")
            if url.startswith(("http://", "https://")) and "<" not in url:
                self.url = url

    def send(self, message: str) -> None:
        print(f"NOTIFY {message}", flush=True)
        if self.url is None:
            return
        parts = urllib.parse.urlsplit(self.url)
        fields = dict(urllib.parse.parse_qsl(parts.query))
        fields.update(title=TITLE, message=message)
        endpoint = urllib.parse.urlunsplit(parts._replace(query=""))
        try:
            urllib.request.urlopen(endpoint, data=urllib.parse.urlencode(fields).encode(), timeout=15).close()
        except Exception as exc:  # noqa: BLE001 — a notification must never stop the baseline
            print(f"WARN  notification failed: {type(exc).__name__}", flush=True)


class Progress:
    """Remaining work of all jobs; thread-safe (one worker thread per GPU)."""

    def __init__(self, plans: dict[tuple[str, str], list[str]], n_gpus: int, notifier: Notifier):
        self.lock = threading.Lock()
        self.remaining = {job: list(stages) for job, stages in plans.items()}
        self.n_jobs = sum(1 for stages in plans.values() if stages)
        self.n_done = 0
        self.n_gpus = max(1, n_gpus)
        self.current: dict[tuple[str, str], tuple[str, float]] = {}
        self.measured: dict[tuple, list[float]] = {}
        self.notifier = notifier

    def _key(self, variant: str, run_name: str, stage: str) -> tuple:
        return (stage, variant, dataset_of(run_name)) if stage == "step2" else (stage, dataset_of(run_name))

    def _estimate(self, variant: str, run_name: str, stage: str) -> float:
        measured = self.measured.get(self._key(variant, run_name, stage))
        if measured:
            return sum(measured) / len(measured)
        dataset = dataset_of(run_name)
        if stage == "step2":
            return DEFAULT_STEP2_S.get(variant, {}).get(dataset, 600)
        if stage.endswith("_test"):
            return DEFAULT_TEST_S
        return DEFAULT_TRAIN_S.get((stage, dataset), 20_000)

    def _remaining_s(self) -> float:
        total = 0.0
        for (variant, run_name), stages in self.remaining.items():
            for stage in stages:
                estimate = self._estimate(variant, run_name, stage) + RENDER_S
                current = self.current.get((variant, run_name))
                if current is not None and current[0] == stage:
                    estimate = max(0.0, estimate - (time.time() - current[1]))
                total += estimate
        return total / self.n_gpus

    def _eta(self) -> str:
        remaining = self._remaining_s()
        if remaining <= 0:
            return "rest 0m"
        finish = datetime.now() + timedelta(seconds=remaining)
        return f"rest ~{format_duration(remaining)} ({finish:%a %H:%M})"

    def started(self) -> None:
        with self.lock:
            message = f"start · {self.n_jobs} runs · {self._eta()}"
        self.notifier.send(message)

    def stage_started(self, variant: str, run_name: str, stage: str) -> None:
        with self.lock:
            self.current[(variant, run_name)] = (stage, time.time())

    def stage_done(self, variant: str, run_name: str, stage: str, seconds: float) -> None:
        with self.lock:
            self.measured.setdefault(self._key(variant, run_name, stage), []).append(seconds)
            stages = self.remaining.get((variant, run_name), [])
            if stage in stages:
                stages.remove(stage)
            self.current.pop((variant, run_name), None)
            message = f"{variant.removeprefix('rwpt-')}/{run_name} {stage} ✓ {format_duration(seconds)} · {self._eta()}"
        self.notifier.send(message)

    def job_done(self, variant: str, run_name: str, status: str, seconds: float, error: str | None = None) -> None:
        if status.startswith("skip"):
            return
        with self.lock:
            self.remaining[(variant, run_name)] = []
            self.current.pop((variant, run_name), None)
            self.n_done += 1
            mark = f"✗ {error[:80]}" if error else f"✓ {format_duration(seconds)}"
            message = (
                f"{variant.removeprefix('rwpt-')}/{run_name} {mark} · run {self.n_done}/{self.n_jobs} · {self._eta()}"
            )
        self.notifier.send(message)

    def all_done(self, seconds: float, n_fail: int) -> None:
        self.notifier.send(f"all done in {format_duration(seconds)} · failures {n_fail}")
