"""Config parse smoke tests."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestConfigParse(unittest.TestCase):
    def test_parse_template_yaml(self):
        from code.utils.yaml_parser import parse_config

        config = parse_config(str(ROOT / "settings" / "template.yaml"))
        self.assertEqual(config.run_configuration.run_name, "template_name")
        self.assertTrue(any("step2" in action for action in config.run_configuration.pipeline))
        self.assertEqual(config.general_configuration.step2.physical_parameters.duration_years, 27.5)


class TestYamlIncludes(unittest.TestCase):
    def test_path_shorthand(self):
        from code.utils.yaml_includes import load_yaml_with_includes

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "injection_temp.yaml").write_text(
                "time_unit: year\nvalues:\n  0.0: 11.0\n  1.0: 11.0\n", encoding="utf-8"
            )
            (root / "step1.yaml").write_text(
                textwrap.dedent(
                    """\
                    datapoints: {train: [0], validation: [1], test: [2]}
                    general: {epochs: 20, visualize: true}
                    """
                ),
                encoding="utf-8",
            )
            (root / "main.yaml").write_text(
                "step1: ./step1.yaml\ninjection_temperature_C: ./injection_temp.yaml\n",
                encoding="utf-8",
            )

            data = load_yaml_with_includes(root / "main.yaml")
            self.assertEqual(data["step1"]["general"]["epochs"], 20)
            self.assertEqual(data["injection_temperature_C"]["values"][0.0], 11.0)

    def test_circular_include_raises(self):
        from code.utils.yaml_includes import load_yaml_with_includes

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.yaml").write_text("x: ./b.yaml\n", encoding="utf-8")
            (root / "b.yaml").write_text("y: ./a.yaml\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_yaml_with_includes(root / "a.yaml")
            self.assertIn("Circular", str(ctx.exception))

    def test_missing_include_raises(self):
        from code.utils.yaml_includes import load_yaml_with_includes

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.yaml").write_text("step1: ./missing.yaml\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                load_yaml_with_includes(root / "main.yaml")


if __name__ == "__main__":
    unittest.main()
