from pathlib import Path
import torch
from torch.nn import MSELoss, L1Loss, HuberLoss
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from utils.utils_args import load_yaml, save_yaml
from preprocessing.data_init import init_data
from preprocessing.transforms import NormalizeTransform
from processing.networks.unetVariants import UNetNoPad2, UNet
from processing.loss_fcts import SSIMLoss, LinfLoss, PATLoss
import utils.logging as log

def preparation(PATH_current_data: Path, PATH_current_model: Path, PATH_destination: Path):
    # load data for dummyK, step1 (predV)

    args = load_yaml(PATH_current_model / "command_line_arguments.yaml")
    args["data_prep"] = PATH_current_data
    args["model"] = PATH_current_model
    args["case"] = "test"
    args["destination"] = PATH_destination
    log.info(args)

    input_channels, output_channels, dataloaders = init_data(args, datapoint_test=args["datapoint_test"], datapoint_validate=args["datapoint_validate"], datapoint_train=args["datapoint_train"], tmp_bool_cutouts=False)
    hparams = load_yaml(PATH_current_model / "HPS_options.yaml") # todo
    log.info(hparams)

    settings = {
      "init_features": hparams["init_features"]["values"][0],
      "depth": hparams["depth"]["values"][0],
      "kernel_size": hparams["kernel_size"]["values"][0],
      "stride": hparams["stride"]["values"][0],
      "dilation": hparams["dilation"]["values"][0],
      "activation": hparams["activation_fct"]["values"][0],
      "norm": hparams["norm"]["values"][0],
      "repeat_inner": hparams["repeat_inner"]["values"][0],
    }
    if "ddunet" in PATH_current_model.name:
        pass
    elif "unetnopad" in PATH_current_model.name:
        model = UNetNoPad2(input_channels, output_channels, **settings)
    elif "unet" in PATH_current_model.name:
        model = UNet(input_channels, output_channels, **settings)
    else:
        model = UNetNoPad2(input_channels, output_channels, **settings)
    model.load(PATH_current_model)
    model.eval()
    
    info = load_yaml(PATH_current_model / "info.yaml")
    norm = NormalizeTransform(info)
    log.info(info)

    return dataloaders, model, norm, info, args, output_channels

def collect_metrics(PATH_current_data, PATH_current_model, PATH_destination, dataloaders, model, norm, info, args, output_channels):
    collected_metrics = {
      "model": PATH_current_model.name,
      "data": PATH_current_data.name,
      "datapoint train": args["datapoint_train"],
      "datapoint test": args["datapoint_test"],
      "datapoint validate": args["datapoint_validate"],
  }
    for case, dataloader in dataloaders.items():
        collected_metrics[case] = {}
        metrics:dict = {"MSE [phys. unit^2]": MSELoss(), "MAE [phys. unit]": L1Loss(), "Linf [phys. unit]": LinfLoss(), "Huber [phys. unit]": HuberLoss(), "SSIM": SSIMLoss()}
        if output_channels == 1: # some only make sense for Temperature predictions
            metrics["PAT0.1 [%]"], metrics["PAT1.0 [%]"] = PATLoss(pat_thresholds=[0.1]), PATLoss(pat_thresholds=[1])
        for metric_name, metric in metrics.items():
            log.info(f"Calculating {metric_name} for {case}",end=" ")
            metrics_values = []

            for batch in dataloader:
                inputs, targets = batch
                outputs = model(inputs).detach()

                inputs, targets = crop_to_output_size(inputs, targets, outputs)
                
                if metric_name in ["SSIM", "IoU"]:
                    values = torch.Tensor([metric(outputs[:,i], targets[:,i]) for i in range(outputs.shape[1])])
                else:
                    # unnormalize inputs and targets
                    reverse_normalization(norm, inputs, targets, outputs)
                    # calc metrics per output channel
                    if "PAT" in metric_name:
                        values = torch.mean(torch.Tensor(metric(outputs, targets).squeeze()))
                    else:
                        values = torch.Tensor([metric(outputs[:,i], targets[:,i]) for i in range(outputs.shape[1])])

                metrics_values.append(values)

            assert len(dataloader) == 1, "I assumed I always have only one batch - otherwise please rethink this code"
            metrics[metric_name] = torch.mean(torch.stack(metrics_values), dim=0) # average over all batches
            log.info(f": average = {metrics[metric_name]}")

            collected_metrics[case][metric_name] = metrics[metric_name]
    save_yaml(collected_metrics, PATH_destination / f"metrics_paper25_{PATH_current_model.name} {PATH_current_data.name}.yaml")

def reverse_normalization(norm, inputs, labels, outputs):
    for tmp_in in inputs:
        norm.reverse(tmp_in, data_type="Inputs")
    for tmp_tar in labels:
        norm.reverse(tmp_tar, data_type="Labels")
    for tmp_out in outputs:
        norm.reverse(tmp_out, data_type="Labels")

def crop_to_output_size(inputs=None, labels=None, outputs=None):
    required_size = outputs.shape[-2:]
    start_pos = ((labels.shape[-2] - required_size[0])//2, (labels.shape[-1] - required_size[1])//2)
    if len(labels.shape) == 4:
        inputs = inputs[:,:,start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]] if inputs is not None else None
        labels = labels[:,:,start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]] if labels is not None else None
    elif len(labels.shape) == 3:
        inputs = inputs[:,start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]] if inputs is not None else None
        labels = labels[:,start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]] if labels is not None else None
    return inputs,labels

def metrics_of_one_model(file_name, n_outputs=2, directory=Path("."), run_ids=[1,2,3,4,5]):
    if "real" in file_name(1):
        metrics = {"train": {}, "val": {}}
        means = {"train": {}, "val": {}}
        stds = {"train": {}, "val": {}}
        mins_maxs = {"train": {}, "val": {}}
    else:
        metrics = {"test": {}, "train": {}, "val": {}}
        means = {"test": {}, "train": {}, "val": {}}
        stds = {"test": {}, "train": {}, "val": {}}
        mins_maxs = {"test": {}, "train": {}, "val": {}}
    inner_metrics = {
      "model": [],
      "Huber": [[] for _ in range(n_outputs)],
      "Linf": [[] for _ in range(n_outputs)],
      "MAE": [[] for _ in range(n_outputs)],
      "MSE": [[] for _ in range(n_outputs)],
      "SSIM": [[] for _ in range(n_outputs)],
    }
    for case in metrics.keys():
        try:
            metrics[case] = deepcopy(inner_metrics)
            if n_outputs == 1:
                metrics[case]["PAT0.1"] = [[]]
                metrics[case]["PAT1.0"] = [[]]

            # load the metrics from the yaml files
            for run in run_ids:
                try:
                    file = directory / file_name(run)
                    tmp = load_yaml(file)
                    metrics[case]["model"].append(tmp["model"])
                    for name, values in tmp[case].items():
                        name = name.split(" ")[0]
                        if n_outputs == 1:
                            metrics[case][name][0].append(values)
                        if n_outputs == 2:
                            metrics[case][name][0].append(values[0])
                            metrics[case][name][1].append(values[1])
                except FileNotFoundError:
                    log.info(f"File {file} not found")
                    continue
            # log.info(metrics_test["Huber"])
            
            # calc mean and std, max and min for each metric in metrics_test for each of the 2 entries
            means_tmp = {}
            stds_tmp = {}
            mins_maxs_tmp = {}
            for name, values in metrics[case].items():
                if name == "model":
                    continue
                means_tmp[name] = []
                stds_tmp[name] = []
                mins_maxs_tmp[name] = [[] for _ in range(n_outputs)]
                for prop in range(n_outputs):
                    means_tmp[name].append(np.mean(values[prop]))
                    stds_tmp[name].append(np.std(values[prop], ddof=1))
                    mins_maxs_tmp[name][prop].append(np.min(values[prop]))
                    mins_maxs_tmp[name][prop].append(np.max(values[prop]))
                    log.info(f"{case} {prop} {name}: {means_tmp[name][prop]:.4f} ± {stds_tmp[name][prop]:.4f} in [{mins_maxs_tmp[name][prop][0]:.4f}, {mins_maxs_tmp[name][prop][1]:.4f}]")
            means[case] = means_tmp
            stds[case] = stds_tmp
            mins_maxs[case] = mins_maxs_tmp
        except:
            log.info(f"{case} does not exist for {file_name}")
            continue
    return metrics, {"means": means, "stds": stds, "mins_maxs": mins_maxs}

def plot_metrics(metrics, title):
    n_cases = len(metrics.keys())
    n_outputs = len(metrics["train"]["MAE"])
    if n_outputs == 1:
        fig, ax0 = plt.subplots(1, 1, figsize=(6, 6))
    elif n_outputs == 2:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        ax0 = axs[0]
    ax0.boxplot(metrics["train"]["MAE"][0], positions=[1], widths=0.5, patch_artist=True, boxprops=dict(facecolor='blue'))
    ax0.boxplot(metrics["val"]["MAE"][0], positions=[2], widths=0.5, patch_artist=True, boxprops=dict(facecolor='green'))
    if n_cases == 3:
        ax0.boxplot(metrics["test"]["MAE"][0], positions=[3], widths=0.5, patch_artist=True, boxprops=dict(facecolor='coral'))
    ax0.boxplot(np.sqrt(metrics["train"]["MSE"][0]), positions=[n_cases+1], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax0.boxplot(np.sqrt(metrics["val"]["MSE"][0]), positions=[n_cases+2], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    if n_cases == 3:
        ax0.boxplot(np.sqrt(metrics["test"]["MSE"][0]), positions=[n_cases+3], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    if n_cases == 3:
        ax0.set_xticks([1, 2, 3, 4, 5, 6])
        ax0.set_xticklabels(["Train MAE", "Val MAE", "Test MAE", "Train RMSE", "Val RMSE", "Test RMSE"])
    else:
        ax0.set_xticks([1, 2, 3, 4])
        ax0.set_xticklabels(["Train MAE", "Val MAE", "Train RMSE", "Val RMSE"])
    if n_outputs == 1:
        ax0.set_ylabel("T [°C]")
    else:
        ax0.set_ylabel("vx [m/y]")
        ax0.set_title("VX Metrics")
    ax0.grid(axis='y', linestyle='--', alpha=0.7)
    if n_outputs == 2:
        axs[1].boxplot(metrics["train"]["MAE"][1], positions=[1], widths=0.5, patch_artist=True, boxprops=dict(facecolor='blue'))
        axs[1].boxplot(metrics["val"]["MAE"][1], positions=[2], widths=0.5, patch_artist=True, boxprops=dict(facecolor='green'))
        if n_cases == 3:
            axs[1].boxplot(metrics["test"]["MAE"][1], positions=[3], widths=0.5, patch_artist=True, boxprops=dict(facecolor='coral'))
        axs[1].boxplot(np.sqrt(metrics["train"]["MSE"][1]), positions=[n_cases+1], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        axs[1].boxplot(np.sqrt(metrics["val"]["MSE"][1]), positions=[n_cases+2], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        if n_cases == 3:
            axs[1].boxplot(np.sqrt(metrics["test"]["MSE"][1]), positions=[n_cases+3], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
        if n_cases == 3:
            axs[1].set_xticks([1, 2, 3, 4, 5, 6])
            axs[1].set_xticklabels(["Train MAE", "Val MAE", "Test MAE", "Train RMSE", "Val RMSE", "Test RMSE"])
        else:
            axs[1].set_xticks([1, 2, 3, 4])
            axs[1].set_xticklabels(["Train MAE", "Val MAE", "Train RMSE", "Val RMSE"])
        axs[1].set_ylabel("vy [m/y]")
        axs[1].set_title("VY Metrics")
        axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    if n_cases == 3:
        plt.legend(["Train MAE", "Val MAE", "Test MAE", "Train RMSE", "Val RMSE", "Test RMSE"], loc="upper left")
    else:
        plt.legend(["Train MAE", "Val MAE", "Train RMSE", "Val RMSE"], loc="upper left")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__=="__main__":

    # calculate metrics for one model
    PATH_PREP_DATA = None
    PATH_MODEL = None
    PATH_DESTINATION = None
    
    dataloaders, model, norm, info, args, output_channels = preparation(PATH_PREP_DATA, PATH_MODEL, PATH_DESTINATION)
    collect_metrics(PATH_PREP_DATA, PATH_MODEL, PATH_DESTINATION, dataloaders, model, norm, info, args, output_channels)

    # exemplary use on how to calculate mean, std, min and max for several runs of one set of hyperparameters
    file_name = lambda run_id: f"metrics_run{run_id}.yaml"
    metrics = metrics_of_one_model(file_name, n_outputs=1, directory=Path("runs") / "modelA_several_runs", run_ids=[1,2,3,4,5])
    plot_metrics(metrics, "Title")