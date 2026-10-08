from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

INCLUDE_KEY = "include"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated with ``override``; nested dicts are merged, everything else replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _include_refs(value: Any) -> list[str | Path]:
    """Normalize the value of an ``include`` key to a list of paths."""
    refs = [value] if isinstance(value, (str, Path)) else value
    if not isinstance(refs, list) or not refs or not all(isinstance(r, (str, Path)) for r in refs):
        raise ValueError(f"'{INCLUDE_KEY}' must be a path or a non-empty list of paths, got {value!r}")
    return refs


def resolve_includes(data: Any, base_dir: Path, *, _stack: tuple[Path, ...] = ()) -> Any:
    """Recursively merge ``include:`` documents into their enclosing mappings."""
    if isinstance(data, list):
        return [resolve_includes(item, base_dir, _stack=_stack) for item in data]
    if not isinstance(data, dict):
        return data

    local = {k: resolve_includes(v, base_dir, _stack=_stack) for k, v in data.items() if k != INCLUDE_KEY}
    if INCLUDE_KEY not in data:
        return local

    included: dict[str, Any] = {}
    for ref in _include_refs(data[INCLUDE_KEY]):
        doc = load_yaml_with_includes(ref, base_dir, _stack=_stack)
        if not isinstance(doc, dict):
            raise ValueError(f"Included YAML {ref!r} must contain a mapping, got {type(doc).__name__}")
        included = _deep_merge(included, doc)  # later includes override earlier ones

    return _deep_merge(included, local)  # local keys override included ones


def load_yaml_with_includes(
    yaml_path: str | Path,
    base_dir: str | Path | None = None,
    *,
    _stack: tuple[Path, ...] = (),
) -> Any:
    """Load a YAML file and recursively resolve ``include:`` directives."""
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    path = Path(yaml_path)
    path = (path if path.is_absolute() else base / path).resolve()

    if path in _stack:
        raise ValueError(f"Circular YAML include: {' -> '.join(str(p) for p in (*_stack, path))}")
    if not path.is_file():
        raise FileNotFoundError(f"YAML include not found: {path} (from {str(yaml_path)!r} relative to {base})")

    with path.open() as f:
        return resolve_includes(yaml.safe_load(f), path.parent, _stack=(*_stack, path))