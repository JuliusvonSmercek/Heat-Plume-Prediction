import argparse
import logging
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _peek_config_device() -> str | None:
    """Read ``run_configuration.device`` from the YAML path in argv (no torch)."""
    from code.utils.yaml_includes import load_yaml_with_includes

    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue
        path = Path(arg)
        if path.suffix.lower() not in {".yaml", ".yml"} or not path.is_file():
            continue
        data = load_yaml_with_includes(path) or {}
        if not isinstance(data, dict):
            continue
        device = (data.get("run_configuration") or {}).get("device")
        return str(device) if device is not None else None
    return None


def _cuda_device_index(device: str | None) -> int | None:
    """Return the physical GPU index for ``cuda`` / ``cuda:N``, else None."""
    if device is None:
        return None
    device = device.strip().lower()
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return None
    return None


def _ensure_runtime_before_torch() -> None:
    """Pin ``CUDA_VISIBLE_DEVICES`` and pip NVIDIA libs before ``torch`` imports.

    ``CUDA_VISIBLE_DEVICES`` must be set before the CUDA runtime initializes, otherwise
    PyTorch still opens an idle ~400 MiB primary context on physical GPU 0 even when
    tensors live on ``cuda:1`` / ``cuda:3``. System CUDA modules on ``LD_LIBRARY_PATH``
    can also conflict with pip-shipped cuDNN, so both fixes re-exec once as needed.
    """
    needs_reexec = False

    if os.environ.get("_HEATPLUME_NVIDIA_LIBS") != "1":
        search_paths: list[str] = list(sys.path)
        if venv := os.environ.get("VIRTUAL_ENV"):
            py = f"python{sys.version_info.major}.{sys.version_info.minor}"
            search_paths.append(str(Path(venv) / "lib" / py / "site-packages"))

        lib_dirs: list[str] = []
        seen: set[str] = set()
        for entry in search_paths:
            nvidia_root = Path(entry) / "nvidia"
            if not nvidia_root.is_dir():
                continue
            for lib_dir in sorted(p for p in nvidia_root.glob("*/lib") if p.is_dir()):
                key = str(lib_dir.resolve())
                if key not in seen:
                    seen.add(key)
                    lib_dirs.append(key)

        os.environ["_HEATPLUME_NVIDIA_LIBS"] = "1"
        if lib_dirs:
            current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
            if current[: len(lib_dirs)] != lib_dirs:
                rest = [p for p in current if p not in lib_dirs]
                os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + rest)
                needs_reexec = True

    if os.environ.get("_HEATPLUME_CUDA_PIN") != "1":
        os.environ["_HEATPLUME_CUDA_PIN"] = "1"
        gpu_index = _cuda_device_index(_peek_config_device())
        if gpu_index is not None:
            desired = str(gpu_index)
            if os.environ.get("CUDA_VISIBLE_DEVICES") != desired:
                os.environ["CUDA_VISIBLE_DEVICES"] = desired
                needs_reexec = True
            os.environ["_HEATPLUME_PHYSICAL_CUDA"] = desired

    if needs_reexec:
        os.execv(sys.executable, [sys.executable, "-m", "code", *sys.argv[1:]])


_ensure_runtime_before_torch()

from code.processing.cnn_main import step_cnn
from code.streamlines.streamline_main import execute_streamline_pipeline
from code.utils import logging as log  # noqa: F401
from code.utils.utils_args import save_yaml
from code.utils.yaml_parser import parse_config

import numpy as np
import torch


def clean_target(config: Any, target: str) -> None:
    """Remove specific artifact directories based on the target identifier."""
    if target == "data_prep":
        path = config.paths.datasets_prep / config.run_configuration.dataset
    elif target in {"results-step1", "results-step2", "results-step3"}:
        path = config.paths.results / config.run_configuration.run_name / target.replace("results-", "")
    else:
        raise ValueError(f"Unknown clean target: {target}")

    if path.exists():
        shutil.rmtree(path)


def set_seed(seed: int, device: str = "cpu") -> None:
    """Set seeds for reproducibility across random, numpy, and torch.

    Only the configured CUDA device is seeded / made current — never ``manual_seed_all``,
    which would open contexts on every visible GPU.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _apply_visible_device_remap(config: Any) -> None:
    """Map YAML ``cuda:N`` → logical ``cuda:0`` after ``CUDA_VISIBLE_DEVICES=N``."""
    physical = os.environ.get("_HEATPLUME_PHYSICAL_CUDA")
    if physical is None:
        return
    if not str(config.run_configuration.device).lower().startswith("cuda"):
        return
    config.run_configuration.device = "cuda:0"
    log.info(f"Using device: cuda:0 (physical cuda:{physical}, CUDA_VISIBLE_DEVICES={physical})")


def main() -> None:
    """Run the LGCNN pipeline from a YAML config.

    ``run_configuration.pipeline`` is an ordered list of actions. Each entry is one of:
    ``clean``, ``step1`` (velocity CNN), ``step2`` (RWPT thermal prior),
    or ``step3`` (temperature CNN). Actions run strictly in listed order; a config may
    include the full sequence or only isolated steps (e.g. only ``step2: run``).
    """
    parser = argparse.ArgumentParser(description="Run the application with a specific configuration.")
    parser.add_argument("config_path", type=str, help="Path to the .yaml configuration file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    config: Any = parse_config(args.config_path)

    run_dir = config.paths.results / config.run_configuration.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_path = run_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log.configure_logging(log_path=log_path, level=log_level)
    log.info(f"Logging to {log_path}")

    _apply_visible_device_remap(config)
    set_seed(config.run_configuration.seed, config.run_configuration.device)

    if config.run_configuration.device != "cpu":
        if os.environ.get("_HEATPLUME_PHYSICAL_CUDA") is None:
            log.info(f"Using device: {config.run_configuration.device}")
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        torch_cuda = torch.version.cuda or ""
        if cuda_home and torch_cuda and Path(cuda_home).name.removeprefix("cuda-").split(".")[0] != torch_cuda.split(".")[0]:
            log.info(
                f"Note: {cuda_home} is loaded, but pip PyTorch uses its own CUDA {torch_cuda}. "
                "You do not need `module load cuda/...` for GPU training in this venv."
            )

    save_yaml(config.model_dump(), run_dir / "config.yaml")

    log.info(f"Run: {config.run_configuration.run_name}, Dataset: {config.run_configuration.dataset}")

    for action in config.run_configuration.pipeline:
        for key, value in action.items():
            match key:
                case "clean":
                    log.info(f"Cleaning target: {value}")
                    clean_target(config, value)
                case "step1":
                    log.info(f"Executing Step 1 in mode: {value}")
                    step_cnn(
                        config.run_configuration,
                        config.paths,
                        config.general_configuration.step1,
                        "step1",
                        value,
                        config.general_configuration.step2.physical_parameters,
                    )
                case "step2":
                    log.info(f"Executing Step 2 in mode: {value}")
                    execute_streamline_pipeline(config, value)
                case "step3":
                    log.info(f"Executing Step 3 in mode: {value}")
                    step_cnn(
                        config.run_configuration,
                        config.paths,
                        config.general_configuration.step3,
                        "step3",
                        value,
                        config.general_configuration.step2.physical_parameters,
                    )


if __name__ == "__main__":
    main()
