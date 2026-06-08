import optuna
from processing.training import training, run
from utils.yaml_parser import AppConfig
from optuna.trial import TrialState
import utils.logging as log

def step_cnn(config: AppConfig, mode: str):
    args = {
      "case": mode,

      "outputs": ''.join(config.general_configuration.step3.model_parameters.outputs),
      "visualize": config.general_configuration.step3.general.visualize,
      "visualize_interval": config.general_configuration.step3.general.visualize_interval,

      "data_prep": config.paths.datasets_prep,
      "data_raw": config.paths.datasets_raw / config.run_configuration.dataset,
      "destination": config.paths.results / config.run_configuration.run_name / "step3",
      "previous_results": config.general_configuration.step3.previous_results,
      
      "epochs": config.general_configuration.step3.general.epochs,

      "datapoint_test": config.general_configuration.step3.datapoints.test,
      "datapoint_validate": config.general_configuration.step3.datapoints.validation,
      "datapoint_train": config.general_configuration.step3.datapoints.train,

      "device": config.run_configuration.device,
      "time_steps_to_predict": config.general_configuration.step3.model_parameters.time_steps_to_predict,
      "max_simulation_timestep": config.general_configuration.step3.model_parameters.max_simulation_timestep,
      "model": config.general_configuration.step3.general.model_path,
      "overfit": config.run_configuration.overfit,
      "overfit_on": config.run_configuration.overfit_on,
    }

    # TODO unused params:
    # - optimizer
    # - activation
    
    args["destination"].mkdir(parents=True, exist_ok=True)

    if mode == "hopt":
        log.error("Hyperparameter optimization not tested once for CNN step yet.")
        hopt = config.general_configuration.step3.hopt_parameters
        args["len_box"] = trial.suggest_categorical("len_box", hopt.len_box)
        args["skip_per_dir"] = trial.suggest_categorical("skip_per_dir", hopt.skip_per_dir)
        args["stride"] = trial.suggest_categorical("stride", hopt.stride)
        args["dilation"] = trial.suggest_categorical("dilation", hopt.dilation)
        args["activation_fct"] = trial.suggest_categorical("activation_fct", hopt.activation)
        args["norm"] = trial.suggest_categorical("norm", hopt.norm)
        args["repeat_inner"] = trial.suggest_categorical("repeat_inner", hopt.repeat_inner)
        args["optimizer_switch"] = trial.suggest_categorical("optimizer_switch", hopt.optimizer_switch)
        args["bool_cutouts"] = trial.suggest_categorical("bool_cutouts", hopt.bool_cutouts)
        args["batchsize"] = trial.suggest_categorical("batchsize", hopt.batchsize)
        args["depth"] = trial.suggest_categorical("depth", hopt.depth)
        args["init_features"] = trial.suggest_categorical("init_features", hopt.init_features)
        args["kernel_size"] = trial.suggest_categorical("kernel_size", hopt.kernel_size)
        args["lr"] = float(trial.suggest_categorical("lr", hopt.lr))
        args["inputs"] = trial.suggest_categorical("inputs", [''.join(item) for item in hopt.inputs])
        args["train_loss"] = trial.suggest_categorical("train_loss", hopt.train_loss)
        args["num_layers"] = trial.suggest_categorical("num_layers", hopt.num_layers)

        log.info("Study name: ", args["destination"])
        study = optuna.create_study(direction="minimize", storage=f"sqlite:///{args["destination"]}/TEST_STUDY.db", study_name="NAME", load_if_exists=True)
        study.optimize(lambda trial: run(trial, args), n_trials=1)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        log.info("Study statistics: ")
        log.info("  Number of finished trials: ", len(study.trials))

        log.info("Best trial:")
        trial = study.best_trial
        log.info("  Value: ", trial.value)

        log.info("  Params: ")
        for key, value in trial.params.items():
            log.info("    {}: {}".format(key, value))
    else:
        parameter = config.general_configuration.step3.model_parameters
        
        # Common parameters for both UNet and RNN
        args["inputs"] = ''.join(parameter.inputs)
        args["train_loss"] = parameter.train_loss
        args["scheduler"] = config.general_configuration.step3.scheduler
        args["network"] = parameter.network
        args["batchsize"] = parameter.batchsize
        args["lr"] = parameter.lr
        args["activation_fct"] = parameter.activation
        args["optimizer_switch"] = parameter.optimizer_switch
        args["bool_cutouts"] = parameter.bool_cutouts
        args["skip_per_dir"] = parameter.skip_per_dir
        args["len_box"] = parameter.len_box
        
        # Network-specific parameters
        if parameter.network == "unet":
            args["len_box"] = parameter.len_box
            args["skip_per_dir"] = parameter.skip_per_dir
            args["stride"] = parameter.stride
            args["dilation"] = parameter.dilation
            args["norm"] = parameter.norm
            args["repeat_inner"] = parameter.repeat_inner
            args["depth"] = parameter.depth
            args["init_features"] = parameter.init_features
            args["kernel_size"] = parameter.kernel_size
            print("UNet parameters loaded for step 3")
            
        elif parameter.network == "rnn":
            args["num_layers"] = parameter.rnn_num_layers
            args["dec_conv_features"] = parameter.dec_conv_features
            args["enc_conv_features"] = parameter.enc_conv_features
            args["enc_kernel_sizes"] = parameter.enc_kernel_sizes
            args["dec_kernel_sizes"] = parameter.dec_kernel_sizes
        
        model = training(args)
