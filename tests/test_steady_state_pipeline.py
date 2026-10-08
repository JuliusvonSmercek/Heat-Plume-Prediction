"""Steady-state pipeline smoke test (settings/test-steady-state-large.yaml)."""

from __future__ import annotations

import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from helpers import ROOT, run_pipeline

SMOKE_STEADY_OUT = Path(__file__).resolve().parent / "heatplume_steady_smoke"


class TestSteadyStatePipelineSmoke(unittest.TestCase):
    def test_steady_state_yaml_minimal_pipeline(self):
        """Run settings/test-steady-state-large.yaml (1 epoch, tiny streamlines).

        Artifacts under tests/heatplume_steady_smoke/ (not cleaned up).
        On CPU, only step1 runs (2560² / 100 HPs is too heavy); full pipeline needs CUDA.
        """
        from code.utils.yaml_parser import parse_config

        config = parse_config(str(ROOT / "settings" / "test-steady-state-large.yaml"))
        raw = Path(config.paths.datasets_raw) / config.run_configuration.dataset
        if not (raw / "RUN_1" / "pflotran.h5").is_file():
            self.skipTest(f"steady-state-large dataset not available under {raw}")

        smoke_root = SMOKE_STEADY_OUT
        smoke_root.mkdir(parents=True, exist_ok=True)
        config.paths.datasets_prep = smoke_root / "datasets_prep"
        config.paths.results = smoke_root / "results"

        # Giant 2560² + 100 HPs: keep full YAML for GPU; step1-only on CPU for wall-time.
        full_pipeline = torch.cuda.is_available() and str(config.run_configuration.device).startswith("cuda")
        if not full_pipeline:
            config.run_configuration.device = "cpu"
            config.run_configuration.pipeline = [
                {"clean": "data_prep"},
                {"clean": "results-step1"},
                {"step1": "train"},
            ]

        def _cpu_safe_dataloader(batchsize: int, dataset, shuffle: bool = True):
            return DataLoader(
                dataset,
                batch_size=min(len(dataset), batchsize),
                shuffle=shuffle,
                drop_last=False,
                num_workers=0,
            )

        with ExitStack() as stack:
            stack.enter_context(patch("code.processing.solver.visualize_outputs"))
            stack.enter_context(patch("code.streamlines.streamline_main.run_visualization"))
            stack.enter_context(
                patch("code.processing.training.construct_dataloader", side_effect=_cpu_safe_dataloader)
            )
            if str(config.run_configuration.device) == "cpu" or not torch.cuda.is_available():
                stack.enter_context(
                    patch(
                        "torch.cuda.get_device_properties",
                        return_value=SimpleNamespace(total_memory=8 * 1024**3, major=8, minor=0),
                    )
                )
                stack.enter_context(patch("torch.cuda.memory_allocated", return_value=0))
                stack.enter_context(patch("torch.cuda.synchronize"))
            try:
                run_pipeline(config)
            except Exception as exc:  # noqa: BLE001
                self.fail(f"steady-state smoke pipeline failed: {exc}")

        run_dir = Path(config.paths.results) / config.run_configuration.run_name
        self.assertTrue((run_dir / "step1" / "model.pt").is_file())
        self.assertTrue((run_dir / "config.yaml").is_file())
        if full_pipeline:
            self.assertTrue((run_dir / "step3" / "model.pt").is_file())


if __name__ == "__main__":
    unittest.main()
