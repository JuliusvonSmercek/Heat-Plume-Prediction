"""Plot spec (npy + yaml) round trip and rendering."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml


class TestPlotSpec(unittest.TestCase):
    def _datapoint(self, dataset_type, prop="Temperature [C]", data=None):
        from code.postprocessing.visualization import DataToVisualize

        if data is None:
            data = torch.linspace(5.0, 16.0, 64 * 16).view(64, 16)
        return DataToVisualize(
            dataset_type,
            data,
            "Label",
            prop,
            (16 * 5.0, 64 * 5.0),
            ambient_temperature_C=10.6,
            temperature_spread_C=5.0,
        )

    def test_round_trip(self):
        from code.plotting.spec import load_panel, write_panel
        from code.preprocessing.datasets.dataset import DatasetType

        dp = self._datapoint(DatasetType.seasonal)
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = write_panel(Path(tmp) / "val_0_output_Temperature [C]_true", dp.data, dp.to_spec("x_true"))
            self.assertEqual(yaml_path.name, "val_0_output_Temperature [C]_true.yaml")
            raw = yaml.safe_load(yaml_path.read_text())
            self.assertEqual(raw["data"], "val_0_output_Temperature [C]_true.npy")
            data, spec = load_panel(yaml_path)
            np.testing.assert_allclose(data, dp.data.numpy(), rtol=1e-6)
            self.assertEqual(spec.colormap.name, "jp_temperature_bidirectional")
            self.assertAlmostEqual(spec.colormap.vmin, 5.6)
            self.assertAlmostEqual(spec.colormap.vmax, 15.6)
            self.assertEqual(spec.axes.extent, [0, 80, 320, 0])

    def test_error_spec_is_free_scaled(self):
        from code.preprocessing.datasets.dataset import DatasetType

        spec = self._datapoint(DatasetType.steady_state_heating).to_spec("Temperature [C]_error")
        self.assertEqual(spec.colormap.name, "jp_linear")
        self.assertIsNone(spec.colormap.vmin)
        self.assertIsNone(spec.colormap.vmax)
        self.assertIsNone(spec.colormap.jp_params)

    def test_unknown_key_rejected(self):
        from code.plotting.spec import load_spec, write_panel
        from code.preprocessing.datasets.dataset import DatasetType

        dp = self._datapoint(DatasetType.seasonal)
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = write_panel(Path(tmp) / "a", dp.data, dp.to_spec("a"))
            raw = yaml.safe_load(yaml_path.read_text())
            raw["axes"]["xlable"] = "typo"
            yaml_path.write_text(yaml.safe_dump(raw))
            with self.assertRaises(ValueError):
                load_spec(yaml_path)

    def test_render_all_cmaps_and_edited_spec(self):
        from code.plotting.spec import write_panel
        from code.plotting.render import render_path
        from code.preprocessing.datasets.dataset import DatasetType

        hp = torch.zeros(64, 16)
        hp[30, 8] = 1.0
        datapoints = {
            "seasonal": self._datapoint(DatasetType.seasonal),
            "heating": self._datapoint(DatasetType.steady_state_heating),
            "cooling": self._datapoint(DatasetType.steady_state_cooling),
            "hp": self._datapoint(DatasetType.seasonal, "Material ID", hp),
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "config.yaml").write_text("run_name: not-a-plot\n")
            for name, dp in datapoints.items():
                write_panel(tmp / name, dp.data, dp.to_spec(name))
            self.assertEqual(datapoints["hp"].to_spec("hp").display.marker_dilation_px, 3)

            # restyle one panel: cutout, no axes, pdf
            raw = yaml.safe_load((tmp / "seasonal.yaml").read_text())
            raw["cutout"] = {"enabled": True, "rows": [10, 40], "cols": [0, 16]}
            raw["axes"]["show"] = False
            raw["output"]["format"] = "pdf"
            (tmp / "seasonal.yaml").write_text(yaml.safe_dump(raw))

            written = render_path(tmp)
            self.assertEqual(sorted(p.name for p in written), ["cooling.png", "heating.png", "hp.png", "seasonal.pdf"])
            for p in written:
                self.assertGreater(p.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
