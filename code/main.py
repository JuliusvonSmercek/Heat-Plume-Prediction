import argparse
import os
import random
import shutil
import sys
from typing import Any, Dict, List

import numpy as np
import torch

from processing.cnn_main import step_cnn
from streamlines.streamline_main import execute_streamline_pipeline
from utils.logging import error, info, set_log_level_debug, set_log_level_info
from utils.utils_args import save_yaml
from utils.yaml_parser import parse_config


def clean_target(config: Any, target: str) -> None:
    """Remove specific artifact directories based on the target identifier."""
    if target == "data_prep":
        path = config.paths.datasets_prep / config.run_configuration.dataset
    elif target in {"results-step1", "results-step2", "results-step3", "results-step4"}:
        path = config.paths.results / config.run_configuration.run_name / target.replace("results-", "")
    else:
        raise ValueError(f"Unknown clean target: {target}")

    if path.exists():
        shutil.rmtree(path)


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    """Main execution entry point handling configuration parsing and pipeline orchestration."""
    parser = argparse.ArgumentParser(description="Run the application with a specific configuration.")
    parser.add_argument("config_path", type=str, help="Path to the .yaml configuration file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    set_log_level_debug() if args.verbose else set_log_level_info()

    config: Any = parse_config(args.config_path)
    set_seed(config.run_configuration.seed)

    if config.run_configuration.device != "cpu":
        info(f"Using device: {config.run_configuration.device} (hint: for gpu support load cuda/12.2.2 module)")

    run_dir = config.paths.results / config.run_configuration.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config.model_dump(), run_dir / "config.yaml")

    info(f"Run: {config.run_configuration.run_name}, Dataset: {config.run_configuration.dataset}")

    for action in config.run_configuration.pipeline:
        for key, value in action.items():
            match key:
                case "clean":
                    info(f"Cleaning target: {value}")
                    clean_target(config, value)
                case "step1":
                    info(f"Executing Step 1 in mode: {value}")
                    error("Step 1 is not implemented yet.")
                case "step2":
                    info(f"Executing Step 2 in mode: {value}")
                    execute_streamline_pipeline(config, value)
                case "step3":
                    info(f"Executing Step 3 in mode: {value}")
                    step_cnn(config, value)


if __name__ == "__main__":
    main()
