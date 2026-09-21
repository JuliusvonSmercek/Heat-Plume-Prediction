from code.processing.training import training
from code.utils import logging as log  # noqa: F401
from code.utils.utils_args import save_yaml
from code.utils.yaml_parser import HoptParameters, MLStepConfig, PhysicalParameters, Paths, RunConfiguration, UNetParameters

import optuna
from optuna.trial import TrialState


def _apply_model_parameter_defaults(args: dict, parameter: UNetParameters) -> None:
    """Fill training args from fixed ``model_parameters`` (non-hopt path)."""
    args["network"] = parameter.network
    args["len_box"] = parameter.len_box
    args["skip_per_dir"] = parameter.skip_per_dir
    args["stride"] = parameter.stride
    args["dilation"] = parameter.dilation
    args["activation_fct"] = parameter.activation
    args["norm"] = parameter.norm
    args["repeat_inner"] = parameter.repeat_inner
    args["optimizer_switch"] = parameter.optimizer_switch
    args["optimizer"] = parameter.optimizer
    args["bool_cutouts"] = parameter.bool_cutouts
    args["batchsize"] = parameter.batchsize
    args["depth"] = parameter.depth
    args["init_features"] = parameter.init_features
    args["kernel_size"] = parameter.kernel_size
    args["inputs"] = "".join(parameter.inputs)
    args["outputs"] = "".join(parameter.outputs)
    args["train_loss"] = parameter.train_loss


def _suggest_hopt_params(trial: optuna.Trial, hopt: HoptParameters) -> dict:
    """Sample one Optuna trial from YAML ``hopt_parameters``."""
    return {
        "network": trial.suggest_categorical("network", hopt.network),
        "len_box": trial.suggest_categorical("len_box", hopt.len_box),
        "skip_per_dir": trial.suggest_categorical("skip_per_dir", hopt.skip_per_dir),
        "stride": trial.suggest_categorical("stride", hopt.stride),
        "dilation": trial.suggest_categorical("dilation", hopt.dilation),
        "activation_fct": trial.suggest_categorical("activation_fct", hopt.activation),
        "norm": trial.suggest_categorical("norm", hopt.norm),
        "repeat_inner": trial.suggest_categorical("repeat_inner", hopt.repeat_inner),
        "optimizer_switch": trial.suggest_categorical("optimizer_switch", hopt.optimizer_switch),
        "optimizer": trial.suggest_categorical("optimizer", hopt.optimizer),
        "bool_cutouts": trial.suggest_categorical("bool_cutouts", hopt.bool_cutouts),
        "batchsize": trial.suggest_categorical("batchsize", hopt.batchsize),
        "depth": trial.suggest_categorical("depth", hopt.depth),
        "init_features": trial.suggest_categorical("init_features", hopt.init_features),
        "kernel_size": trial.suggest_categorical("kernel_size", hopt.kernel_size),
        "inputs": trial.suggest_categorical("inputs", ["".join(item) for item in hopt.inputs]),
        "outputs": trial.suggest_categorical("outputs", ["".join(item) for item in hopt.outputs]),
        "train_loss": trial.suggest_categorical("train_loss", hopt.train_loss),
    }


def _trial_folder_name(trial_number: int, params: dict) -> str:
    """Readable per-trial directory under step3/ (keeps all combinations inspectable)."""
    parts = [f"trial_{trial_number:04d}"]
    for key, prefix in (
        ("depth", "d"),
        ("init_features", "f"),
        ("kernel_size", "k"),
        ("batchsize", "bs"),
        ("norm", "n"),
        ("train_loss", "loss"),
    ):
        if key in params and params[key] is not None:
            parts.append(f"{prefix}{params[key]}")
    return "_".join(parts)


def step_cnn(
    run_configuration: RunConfiguration,
    paths: Paths,
    step_config: MLStepConfig,
    step_name: str,
    mode: str,
    physical_parameters: PhysicalParameters,
):
    args = {
        "case": mode,
        "model": paths.results / run_configuration.run_name / step_name,
        "visualize": step_config.general.visualize,
        "visualize_epochs": step_config.general.visualize_epochs,
        "data_prep": paths.datasets_prep,
        "data_raw": paths.datasets_raw / run_configuration.dataset,
        "destination": paths.results / run_configuration.run_name / step_name,
        "epochs": step_config.general.epochs,
        "datapoint_test": step_config.datapoints.test,
        "datapoint_validate": step_config.datapoints.validation,
        "datapoint_train": step_config.datapoints.train,
        "device": run_configuration.device,
        "scheduler": step_config.scheduler,
        "ambient_temperature_C": physical_parameters.ambient_temperature_C,
        "temperature_spread_C": physical_parameters.temperature_spread_C,
    }

    args["destination"].mkdir(parents=True, exist_ok=True)

    if mode == "hopt":
        if step_config.hopt_parameters is None:
            raise ValueError(f"{step_name}: mode 'hopt' requires general_configuration.{step_name}.hopt_parameters")
        if not isinstance(step_config.model_parameters, UNetParameters):
            raise ValueError(f"{step_name}: hopt currently supports network: unet model_parameters only")

        hopt = step_config.hopt_parameters
        study_dir = args["destination"]
        log.info(f"Optuna study directory: {study_dir}")
        study = optuna.create_study(
            direction="minimize",
            storage=f"sqlite:///{study_dir}/optuna_study.db",
            study_name=f"{run_configuration.run_name}_{step_name}",
            load_if_exists=True,
        )

        def run_hopt(trial: optuna.Trial) -> float:
            args_copy = args.copy()
            args_copy["case"] = "train"
            # Keep YAML visualize so each trial can dump val/test pics under its folder
            args_copy["visualize"] = step_config.general.visualize

            suggested = _suggest_hopt_params(trial, hopt)
            args_copy.update(suggested)

            trial_dir = study_dir / _trial_folder_name(trial.number, suggested)
            args_copy["destination"] = trial_dir
            args_copy["model"] = trial_dir
            trial_dir.mkdir(parents=True, exist_ok=True)

            save_yaml({"trial_number": trial.number, **suggested}, trial_dir / "trial_params.yaml")
            log.info(f"Trial {trial.number} → {trial_dir.name}")

            val_loss = training(args_copy)
            save_yaml(
                {"trial_number": trial.number, "val_loss": val_loss, **suggested},
                trial_dir / "trial_result.yaml",
            )
            return float(val_loss)

        study.optimize(run_hopt, n_trials=hopt.n_trials)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        log.info("Study statistics: ")
        log.info(f"  Pruned trials: {len(pruned_trials)}")
        log.info(f"  Complete trials: {len(complete_trials)}")
        log.info(f"  Number of finished trials: {len(study.trials)}")

        if study.best_trial is not None:
            best = study.best_trial
            log.info("Best trial:")
            log.info(f"  Value: {best.value}")
            log.info("  Params:")
            for key, value in best.params.items():
                log.info(f"    {key}: {value}")
            save_yaml(
                {
                    "trial_number": best.number,
                    "val_loss": best.value,
                    "params": best.params,
                    "folder": _trial_folder_name(best.number, best.params),
                },
                study_dir / "best_trial.yaml",
            )
    else:
        parameter = step_config.model_parameters
        if not isinstance(parameter, UNetParameters):
            raise ValueError(f"{step_name}: expected unet model_parameters for mode '{mode}'")
        _apply_model_parameter_defaults(args, parameter)
        training(args)
