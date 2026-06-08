import gc
import logging
import multiprocessing
import numpy as np
import torch
from torch.nn import MSELoss, L1Loss, HuberLoss
from datetime import datetime
from typing import Dict
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
import utils.logging as log

from preprocessing.preprocessing import preprocessing
from preprocessing.data_init import construct_dataloader, init_data
from processing.networks.unetVariants import UNetNoPad2, UNet
from processing.networks.convLSTM import Seq2Seq
from processing.solver import Solver
from processing.loss_fcts import CombiLoss, CombinedFocalMSE, FocalMAE, WeightedMSE, SSIMLoss, FocalMSE, CustomLoss, BinaryCrossEntropy, FocalMSE_FP, BCEMAELoss, MAE_Logits, FocalMAE_FP
from postprocessing.visualization import visualize_inputs, visualize_outputs, visualize_outputs_over_inputs
from utils.utils_args import get_data_prep_path, load_yaml, save_yaml, check_model_avail, make_data_prep_dir, load_time_steps
from preprocessing.validity_checks import receptive_field_is_sufficient

def save_predictions(model, dataset, result_destination: Path, args: Dict):
    """Run inference on dataset and save predictions as individual .pt files"""
    device = args["device"]
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Run inference
        if dataset.__class__.__name__ in ["SimulationDatasetCutsSequential", "DataPointSequence", "Subset"]:
            if inputs.shape[-1] > 500 or inputs.shape[-2] > 500:
                y_out = model.infer_tiled(inputs, device)
            else:
                y_out = model.infer(inputs, device)
        else:
            y_out = model.infer(inputs, device)
        
        # Save individual prediction and preserve run id naming
        prediction = y_out.detach().cpu().squeeze(0)
        if hasattr(dataset, 'input_names') and batch_idx < len(dataset.input_names):
            filename = Path(dataset.input_names[batch_idx]).name
        else:
            filename = f"RUN_{batch_idx}.pt"
        torch.save(prediction, result_destination / filename)
        
        if batch_idx % 10 == 0:
            log.info(f"Saved prediction {batch_idx + 1}/{len(dataloader)} as {filename}")
    
    log.info(f"Saved {len(dataloader)} predictions to {result_destination}")

def training(args: Dict):
    multiprocessing.set_start_method("spawn", force=True)
    log.configure_logging(log_path=args["destination"] / "training.log", level=logging.INFO, clear_handlers=False)

    args["data_prep"] = get_data_prep_path(args["data_prep"], args["inputs"], args["outputs"], args["data_raw"])
    preprocessing(args) # and save info.yaml in model folder
    log.info("Datapoint test: args['datapoint_test']: {}".format(args["datapoint_test"]))
    log.info("Datapoint validate: args['datapoint_validate']: {}".format(args["datapoint_validate"]))
    input_channels, output_channels, datasets = init_data(args, datapoint_test=args["datapoint_test"], datapoint_validate=args["datapoint_validate"], datapoint_train=args["datapoint_train"], tmp_bool_cutouts=args["bool_cutouts"],log_path=args["destination"] / "training.log")

    dataloaders = {}
    dataloaders["train"] = construct_dataloader(args["batchsize"], datasets["train"], shuffle=True)
    log.info(f"Datapoints in dataset_train: {len(dataloaders['train'].dataset)}")
    dataloaders["val"] = construct_dataloader(1, datasets["val"], shuffle=True)
    #dataloaders["val"] =dataloaders["train"] # TODO
    #log.info(f"Datapoints in dataset_val: {len(dataloaders['val'].dataset)}")
    dataloaders["test"] = construct_dataloader(1, datasets["test"],shuffle=True)
    #dataloaders["test"] = dataloaders["train"] # TODO
    log.info(f"Datapoints in dataset_test: {len(dataloaders['test'].dataset)}")

    # Step 1
    if "t" not in args["outputs"]:
        model = UNet(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"]).float()
        log.info("Using UNet for step 1")
    # Step 3
    else:
        network = args.get("network", "unet").lower()
        if network in ["convlstm", "rnn", "lstm"]:
            print("Using ConvLSTM for current step")
            if "time_steps_to_predict" in args and args["time_steps_to_predict"] is not None:
                time_steps_to_predict = args["time_steps_to_predict"]
            else:
                time_steps_to_predict = load_time_steps(Path(args["data_raw"],"RUN_"+str(args["order_data"][0]), "pflotran.h5"))
            # Support nested lists for sublist scheduling
            if isinstance(time_steps_to_predict, list) and len(time_steps_to_predict) > 0 and isinstance(time_steps_to_predict[0], list):
                if len(set(len(sub) for sub in time_steps_to_predict)) != 1:
                    raise ValueError("All time_steps_to_predict sublists must have the same length")
                extend = len(time_steps_to_predict[0])
            else:
                extend = len(time_steps_to_predict)

            log.info(f"Extend: {extend}")
            len_box = args["len_box"]
            print(f"Input_channels before net construction: {input_channels}")
            model = Seq2Seq(input_channels+2,
                            frame_size=[len_box, len_box],
                            prev_boxes=0,
                            extend=extend,
                            num_layers=args["num_layers"],
                            enc_conv_features=args["enc_conv_features"],
                            dec_conv_features=args["dec_conv_features"],
                            enc_kernel_sizes=args["enc_kernel_sizes"],
                            dec_kernel_sizes=args["dec_kernel_sizes"],
                            ).float()
            proj = model.sequential[0].convLSTMcell.proj
            receptive_field_dict = model.calculate_receptive_field()
            if not receptive_field_is_sufficient(receptive_field_dict['total_rf'], dataloaders):
                log.warning(f"Warning: ConvLSTM receptive field {receptive_field_dict['total_rf']} is smaller than the required input size.")
        else:
            log.info("Using UNet for step 3")
            model = UNetNoPad2(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"], stride=args["stride"], dilation=args["dilation"], activation=args["activation_fct"], norm=args["norm"], repeat_inner=args["repeat_inner"]).float()

    #model = nn.DataParallel(model)
    model.to(args["device"])
    loss = select_loss_function(args)
    log.info(f"Loss function selected: {loss.__class__.__name__}")
    solver = Solver(model, datasets["train"], datasets["val"], loss_func=loss, finetune=(args["case"] == "finetune"), optimizer_switch=args["optimizer_switch"], learning_rate=args["lr"], batchsize=args["batchsize"])
    if args["case"] in ["test", "finetune"]:
        check_model_avail(args)
        model.load(args["model"], args["device"])
    if args["case"] == "test":
        model.eval()
        log.info(f"Number of test datapoints: {len(dataloaders['test'].dataset)}")
        solver.save_metrics_separate_yaml(dataloaders["train"], dataloaders["val"], dataloaders["test"], args["destination"], model.num_of_params(), args["epochs"], None, args["device"])
        

    if args["case"] in ["train", "finetune"]:
        
        # TODO check if correct val-loss in solver (line 73), depends on real or dummy k (permeability)
        training_time = datetime.now()
        try:
            solver.train(dataloaders["train"], dataloaders["val"], args)
        except KeyboardInterrupt:
            if solver.best_model_params is not None:
                log.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
            else:
                log.warning("Manually stopping training early. No best model found yet.")
        finally:
            log.info("Training finished")

        # save model 
        training_time = datetime.now() - training_time
        model.save(args["destination"])
        solver.save_metrics_separate_yaml(dataloaders["train"], dataloaders["val"], dataloaders["test"],args["destination"], model.num_of_params(), args["epochs"], training_time.total_seconds(), args["device"])

    # postprocessing, visualize only outputs
    if args["visualize"]:
        
        visualize_outputs(model, dataloaders["test"], args, plot_path=args["destination"] / "test", amount_datapoints_to_visu=2, pic_format="png")
        visualize_inputs(dataloaders["test"], args, amount_datapoints_to_visu=2, plot_path=args["destination"] / "val", pic_format="png")
    
    # save results  
    result_destination = args["destination"] / "results"
    result_destination.mkdir(parents=True, exist_ok=True)
    
    # Run inference and save predictions
    # model.eval()
    # with torch.no_grad():
    #     #Save predictions for full training dataset
    #     log.info("Running inference on full training dataset and saving results...")
    #     save_predictions(model, datasets["train_full_dp"], result_destination, args)
        

    
    # Clean up dataloaders
    for key in ["train", "val", "test"]:
      if hasattr(dataloaders[key], '_iterator') and dataloaders[key]._iterator is not None:
        try:
          dataloaders[key]._iterator._shutdown_workers()
        except Exception:
          pass # Prevent error masking if they are already dead
      del dataloaders[key]
    gc.collect()

    return model


# only used for hopt
def run(trial, args: Dict):
    (args["destination"] / "models").mkdir(parents=True, exist_ok=True)

    run_name = trial.number

    multiprocessing.set_start_method("spawn", force=True)
    args["data_prep"] = get_data_prep_path(args["data_prep"], args["inputs"], args["outputs"], args["data_raw"])

    save_yaml(args, args["destination"] / "command_line_arguments.yaml")

    # data
    preprocessing(args) # and save info.yaml in model folder
    input_channels, output_channels, datasets = init_data(args, datapoint_test=args["datapoint_test"], datapoint_validate=args["datapoint_validate"], datapoint_train=args["datapoint_train"], tmp_bool_cutouts=args["bool_cutouts"])

    try:
        # model
        model = UNetNoPad2(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"], stride=args["stride"], dilation=args["dilation"], activation=args["activation_fct"], norm=args["norm"], repeat_inner=args["repeat_inner"]).float()
        model.to(args["device"])
        
        if args["case"] in ["test", "finetune"]:
            check_model_avail(args)
            model.load(args["model"], args["device"])
        if args["case"] == "test":
            model.eval()

        if args["case"] in ["train", "finetune"]:
            loss_mapping = {
                "mae": L1Loss(),
                "mse": MSELoss()
            }
            loss = loss_mapping.get(args["train_loss"].lower(), MSELoss())
            solver = Solver(model, datasets["train"], datasets["val"], loss_func=loss, finetune=(args["case"] == "finetune"), optimizer_switch=args["optimizer_switch"], learning_rate=float(args["lr"]))
            try:
                # solver.load_lr_schedule(args["destination"] / "learning_rate_history.csv")
                val_loss = solver.train(args, optuna_trial=trial)
            except KeyboardInterrupt:
                logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
                val_loss = solver.best_model_params["loss"]

            # save model 
            model.save(args["destination"] / "models", f"{run_name}.pt")

    except Exception as e:
        log.info(f"An error occurred: {e}")
        val_loss = 0.2

    # Clear up memory
    del model
    del datasets
    torch.cuda.empty_cache()

    return val_loss

def select_loss_function(args):
    if args["train_loss"].lower() == "mae":
        loss = L1Loss()
    elif args["train_loss"].lower() == "mse":
        loss = MSELoss()
    elif args["train_loss"].lower() == "weightedmse":
        loss = WeightedMSE()
    elif args["train_loss"].lower() == "huber":
        loss = HuberLoss()
    elif args["train_loss"].lower() == "combi":
        loss = CombiLoss(0.75)
    elif args["train_loss"].lower() == "ssim":
        loss = SSIMLoss()
    elif args["train_loss"].lower() == "focalmse":
        loss = FocalMSE()
    elif args["train_loss"].lower() == "combined_focalmse":
        loss = CombinedFocalMSE()
    elif args["train_loss"].lower() == "custom":
        loss = CustomLoss()
    elif args["train_loss"].lower() == "bce":
        loss = BinaryCrossEntropy()
    elif args["train_loss"].lower() == "focalmse_fp":
        loss = FocalMSE_FP()
    elif args["train_loss"].lower() == "focalmae_fp":
        loss = FocalMAE_FP()
    elif args["train_loss"].lower() == "focalmae":
        loss = FocalMAE()
    elif args["train_loss"].lower() == "bce_f1_loss":
        loss = BCEMAELoss()
    elif args["train_loss"].lower() == "mae_logits":
        loss = MAE_Logits()
    return loss
