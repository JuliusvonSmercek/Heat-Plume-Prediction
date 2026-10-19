"""Hierarchical wall-clock timing for the whole pipeline, dumped to ``timing.yaml``.

Usage::

    from code.utils.timing import TIMER

    with TIMER.section("training"):
        ...
        TIMER.note("epochs_run", 120)

Sections nest by call stack. A section entered several times under the same parent
(e.g. one per epoch or per datapoint) is aggregated into one node with ``calls``,
``mean_s``, ``min_s`` and ``max_s``. ``other_s`` is the part of a node's time not
covered by its child sections. With ``enable_cuda_sync`` every section boundary calls
``torch.cuda.synchronize()`` so asynchronous GPU work is charged to the right section.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

_DIGITS = 4


@dataclass
class _Node:
    total_s: float = 0.0
    calls: int = 0
    min_s: float = float("inf")
    max_s: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)
    children: dict[str, "_Node"] = field(default_factory=dict)

    def add(self, seconds: float) -> None:
        self.total_s += seconds
        self.calls += 1
        self.min_s = min(self.min_s, seconds)
        self.max_s = max(self.max_s, seconds)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"total_s": round(self.total_s, _DIGITS)}
        if self.calls > 1:
            out["calls"] = self.calls
            out["mean_s"] = round(self.total_s / self.calls, _DIGITS)
            out["min_s"] = round(self.min_s, _DIGITS)
            out["max_s"] = round(self.max_s, _DIGITS)
        out.update(self.notes)
        if self.children:
            covered = sum(child.total_s for child in self.children.values())
            out["other_s"] = round(max(self.total_s - covered, 0.0), _DIGITS)
            for name, child in self.children.items():
                out[name] = child.to_dict()
        return out


class Timer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._root = _Node()
        self._stack: list[_Node] = [self._root]
        self._cuda_sync = False
        self._started_at = time.time()
        self._start = time.perf_counter()

    def enable_cuda_sync(self, enabled: bool = True) -> None:
        self._cuda_sync = enabled

    def _sync(self) -> None:
        if not self._cuda_sync:
            return
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        parent = self._stack[-1]
        node = parent.children.setdefault(name, _Node())
        self._stack.append(node)
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            node.add(time.perf_counter() - start)
            self._stack.pop()

    def note(self, key: str, value: Any) -> None:
        """Attach a value (e.g. epoch count, sample count) to the current section."""
        self._stack[-1].notes[key] = value

    def to_dict(self, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        total = time.perf_counter() - self._start
        out: dict[str, Any] = dict(meta or {})
        out["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self._started_at))
        out["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        out["total_s"] = round(total, _DIGITS)
        out.update(self._root.notes)
        covered = sum(child.total_s for child in self._root.children.values())
        out["other_s"] = round(max(total - covered, 0.0), _DIGITS)
        for name, child in self._root.children.items():
            out[name] = child.to_dict()
        return out

    def save(self, path: str | Path, meta: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(meta), sort_keys=False), encoding="utf-8")
        return path


TIMER = Timer()
