"""Optuna hopt wiring: YAML → trial folders → solver args."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class TestResolveOptimizer(unittest.TestCase):
    def test_known_names(self):
        from torch.optim import SGD, Adam, AdamW

        from code.processing.solver import resolve_optimizer

        self.assertIs(resolve_optimizer("AdamW"), AdamW)
        self.assertIs(resolve_optimizer("adam"), Adam)
        self.assertIs(resolve_optimizer("SGD"), SGD)

    def test_unknown_raises(self):
        from code.processing.solver import resolve_optimizer

        with self.assertRaises(ValueError):
            resolve_optimizer("LBFGS")


class TestHoptConfigParse(unittest.TestCase):
    def test_template_hopt_has_n_trials(self):
        from code.utils.yaml_parser import parse_config

        config = parse_config(str(ROOT / "settings" / "template.yaml"))
        self.assertEqual(config.general_configuration.step3.hopt_parameters.n_trials, 100)
        self.assertIn(3, config.general_configuration.step3.hopt_parameters.kernel_size)

    def test_seasonal_pflotran_hopt_space(self):
        from code.utils.yaml_parser import parse_config

        config = parse_config(str(ROOT / "settings" / "seasonal_pflotran-dataset.yaml"))
        hopt = config.general_configuration.step3.hopt_parameters
        self.assertIsNotNone(hopt)
        self.assertEqual(hopt.n_trials, 40)
        self.assertEqual(hopt.kernel_size, [3, 5, 7])
        self.assertEqual(hopt.depth, [2, 3, 4])
        self.assertEqual(hopt.init_features, [32, 64, 128])
        self.assertTrue(any(action.get("step3") == "hopt" for action in config.run_configuration.pipeline))


class TestHoptTrialFolders(unittest.TestCase):
    def test_each_trial_gets_own_folder_and_params(self):
        from code.processing.cnn_main import step_cnn
        from code.utils.yaml_parser import parse_config

        config = parse_config(str(ROOT / "settings" / "seasonal_pflotran-dataset.yaml"))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config.paths.results = tmp_path / "results"
            config.paths.datasets_prep = tmp_path / "prep"
            config.paths.datasets_raw = tmp_path / "raw"
            config.run_configuration.run_name = "hopt_unit"
            config.general_configuration.step3.hopt_parameters.n_trials = 2
            config.general_configuration.step3.general.visualize = False

            seen_destinations: list[Path] = []

            def fake_training(args: dict) -> float:
                seen_destinations.append(Path(args["destination"]))
                # Assert solver-facing knobs are present for every trial
                for key in (
                    "depth",
                    "init_features",
                    "kernel_size",
                    "optimizer",
                    "network",
                    "batchsize",
                    "train_loss",
                    "activation_fct",
                    "norm",
                    "stride",
                    "dilation",
                    "repeat_inner",
                    "bool_cutouts",
                    "len_box",
                    "skip_per_dir",
                    "inputs",
                    "outputs",
                ):
                    self.assertIn(key, args, msg=f"missing {key}")
                self.assertTrue(args["destination"].is_dir())
                self.assertTrue((args["destination"] / "trial_params.yaml").is_file())
                return 0.42 + len(seen_destinations) * 0.01

            with patch("code.processing.cnn_main.training", side_effect=fake_training):
                step_cnn(
                    config.run_configuration,
                    config.paths,
                    config.general_configuration.step3,
                    "step3",
                    "hopt",
                    config.general_configuration.step2.physical_parameters,
                )

            step3 = config.paths.results / config.run_configuration.run_name / "step3"
            self.assertEqual(len(seen_destinations), 2)
            self.assertEqual(len({d.resolve() for d in seen_destinations}), 2)
            for dest in seen_destinations:
                self.assertTrue(dest.name.startswith("trial_"))
                self.assertTrue((dest / "trial_result.yaml").is_file())
            self.assertTrue((step3 / "optuna_study.db").is_file())
            self.assertTrue((step3 / "best_trial.yaml").is_file())


class TestTrialFolderName(unittest.TestCase):
    def test_includes_architecture_knobs(self):
        from code.processing.cnn_main import _trial_folder_name

        name = _trial_folder_name(7, {"depth": 3, "init_features": 64, "kernel_size": 5, "batchsize": 20})
        self.assertEqual(name, "trial_0007_d3_f64_k5_bs20")


if __name__ == "__main__":
    unittest.main()
