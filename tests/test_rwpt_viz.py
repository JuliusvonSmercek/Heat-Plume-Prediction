"""RWPT intermediate-stage visualization smoke test."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


class TestRwptViz(unittest.TestCase):
    def _write(self, out_dir: Path, render: bool) -> Path:
        from code.postprocessing.visualization import visualize_rwpt_fields

        h, w = 16, 32
        temp = torch.full((h, w), 10.6)
        temp[8, 16] = 16.0
        acc = torch.ones((h, w))
        acc[8, 16] = 0.4
        velocity = torch.stack(
            [torch.ones((h, w)), torch.zeros((h, w))]
        ).unsqueeze(0)
        wells = torch.tensor([[8.5, 16.5]])
        return visualize_rwpt_fields(
            out_dir / "L00_16x32_it00.png",
            "RWPT[L0(16x32)] iter 0",
            ambient_temp_C=10.6,
            min_temp_C=10.6,
            max_temp_C=15.6,
            temp_spread_C=5.0,
            resolution_m_per_px=40.0,
            temp_C=temp,
            acceptance=acc,
            acceptance_next=acc * 0.8,
            velocity_m_per_year=velocity,
            wells_rowcol_px=wells,
            render=render,
        )

    def test_visualize_rwpt_fields_writes_specs_only(self):
        from code.plotting.spec import load_panel, load_spec

        with tempfile.TemporaryDirectory() as tmp:
            figure_path = self._write(Path(tmp), render=False)
            self.assertEqual(figure_path.name, "L00_16x32_it00.figure.yaml")
            figure = load_spec(figure_path)
            self.assertEqual(
                [Path(p).stem for p in figure.panels],
                [f"L00_16x32_it00_{s}" for s in ("speed", "acceptance", "acceptance_next", "temperature", "overshoot")],
            )
            data, spec = load_panel(Path(tmp) / "L00_16x32_it00_temperature.yaml")
            self.assertEqual(data.shape, (16, 32))
            self.assertAlmostEqual(float(data[8, 16]), 16.0, places=5)
            self.assertEqual(spec.colormap.name, "jp_temperature_upperlinear")
            self.assertEqual(spec.overlays.wells_rowcol_px, [[8.5, 16.5]])
            self.assertEqual(list(Path(tmp).glob("*.png")), [])

    def test_visualize_rwpt_fields_writes_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(Path(tmp), render=True)
            out = Path(tmp) / "L00_16x32_it00.png"
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 1000)

    def test_as_numpy_hw_collapses_velocity_stack(self):
        from code.postprocessing.visualization import as_numpy_hw

        field = torch.zeros((1, 2, 4, 5))
        field[0, 0, 1, 2] = 3.0
        field[0, 1, 1, 2] = 4.0
        out = as_numpy_hw(field)
        self.assertEqual(out.shape, (4, 5))
        self.assertAlmostEqual(float(out[1, 2]), 5.0)

    def test_signed_overshoot(self):
        from code.postprocessing.visualization import signed_overshoot_C

        temp = np.array([[12.0, 16.0], [4.0, 10.0]])
        signed = signed_overshoot_C(temp, min_temp_C=5.0, max_temp_C=15.0)
        np.testing.assert_allclose(signed, np.array([[0.0, 1.0], [-1.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
