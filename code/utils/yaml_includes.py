"""Resolve YAML path includes (``.yaml`` / ``.yml`` values → nested documents)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_YAML_SUFFIXES = {".yaml", ".yml"}


def resolve_includes(data: Any, base_dir: Path, *, _stack: tuple[Path, ...] = ()) -> Any:
    """Replace ``*.yaml`` / ``*.yml`` strings with the loaded document (paths relative to ``base_dir``)."""
    if isinstance(data, dict):
        return {k: resolve_includes(v, base_dir, _stack=_stack) for k, v in data.items()}
    if isinstance(data, list):
        return [resolve_includes(item, base_dir, _stack=_stack) for item in data]
    if isinstance(data, str) and Path(data).suffix.lower() in _YAML_SUFFIXES:
        return load_yaml_with_includes(data, base_dir, _stack=_stack)
    return data


def load_yaml_with_includes(
    yaml_path: str | Path,
    base_dir: str | Path | None = None,
    *,
    _stack: tuple[Path, ...] = (),
) -> Any:
    """Load a YAML file and recursively resolve path includes."""
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    path = Path(yaml_path)
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()

    if path in _stack:
        raise ValueError(f"Circular YAML include: {' -> '.join(str(p) for p in (*_stack, path))}")
    if not path.is_file():
        raise FileNotFoundError(f"YAML include not found: {path} (from {yaml_path!r} relative to {base})")

    with path.open() as f:
        return resolve_includes(yaml.safe_load(f), path.parent, _stack=(*_stack, path))
