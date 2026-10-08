"""Seasonal pipeline smoke test (settings/test-seasonal.yaml)."""

from __future__ import annotations

import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from helpers import ROOT, run_pipeline

SMOKE_SEASONAL_OUT = Path(__file__).resolve().parent / "heatplume_seasonal_smoke"


class TestSeasonalPipelineSmoke(unittest.TestCase):
    def test_seasonal_yaml_minimal_pipeline(self):
        """Run settings/test-seasonal.yaml (1 epoch, 2 RUNs, tiny streamlines).

        Full step1→2→3 runs on CPU with the smoke settings (1000², few HPs).
        """
        from code.utils.yaml_parser import parse_config

        config = parse_config(str(ROOT / "settings" / "test-seasonal.yaml"))
        raw = Path(config.paths.datasets_raw) / config.run_configuration.dataset
        if not (raw / "RUN_0" / "pflotran.h5").is_file():
            self.skipTest(f"seasonal dataset not available under {raw}")

        smoke_root = SMOKE_SEASONAL_OUT
        smoke_root.mkdir(parents=True, exist_ok=True)
        config.paths.datasets_prep = smoke_root / "datasets_prep"
        config.paths.results = smoke_root / "results"

        if not torch.cuda.is_available():
            config.run_configuration.device = "cpu"

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
                self.fail(f"seasonal smoke pipeline failed: {exc}")

        run_dir = Path(config.paths.results) / config.run_configuration.run_name
        self.assertTrue((run_dir / "step1" / "model.pt").is_file())
        self.assertTrue((run_dir / "config.yaml").is_file())
        self.assertTrue((run_dir / "step3" / "model.pt").is_file())


if __name__ == "__main__":
    unittest.main()
