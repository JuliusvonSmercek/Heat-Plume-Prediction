"""Hierarchical pipeline timer."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import yaml


class TestTimer(unittest.TestCase):
    def test_nested_aggregated_sections(self):
        from code.utils.timing import Timer

        timer = Timer()
        with timer.section("step1_train"):
            with timer.section("training"):
                timer.note("epochs_run", 3)
                for _ in range(3):
                    with timer.section("epoch_train_pass"):
                        time.sleep(0.01)
            with timer.section("evaluation"):
                with timer.section("test"):
                    for _ in range(2):
                        with timer.section("model_forward"):
                            time.sleep(0.005)

        with tempfile.TemporaryDirectory() as tmp:
            path = timer.save(Path(tmp) / "timing.yaml", meta={"run_name": "unit"})
            out = yaml.safe_load(path.read_text())

        self.assertEqual(out["run_name"], "unit")
        step1 = out["step1_train"]
        epochs = step1["training"]["epoch_train_pass"]
        self.assertEqual(step1["training"]["epochs_run"], 3)
        self.assertEqual(epochs["calls"], 3)
        self.assertGreaterEqual(epochs["total_s"], 0.03)
        self.assertAlmostEqual(epochs["mean_s"], epochs["total_s"] / 3, places=3)
        forward = step1["evaluation"]["test"]["model_forward"]
        self.assertEqual(forward["calls"], 2)
        self.assertGreaterEqual(step1["total_s"], epochs["total_s"] + forward["total_s"])
        self.assertIn("other_s", step1)
        self.assertGreaterEqual(out["total_s"], step1["total_s"])

    def test_section_records_time_on_exception(self):
        from code.utils.timing import Timer

        timer = Timer()
        with self.assertRaises(RuntimeError):
            with timer.section("step2_run"):
                raise RuntimeError("boom")
        self.assertIn("step2_run", timer.to_dict())


if __name__ == "__main__":
    unittest.main()
