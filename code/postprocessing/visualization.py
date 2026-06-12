from code.postprocessing.cmap_jp import *  # noqa: F403
from code.preprocessing.data_init import _label_has_midcell_above_threshold
from code.preprocessing.datasets.dataset import DatasetType
from code.preprocessing.transforms import NormalizeTransform
from code.processing.networks.unetVariants import UNet
from code.utils import logging as log  # noqa: F401
from copy import deepcopy
from dataclasses import dataclass, field
from math import inf

import matplotlib
import numpy as np
import torch
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use("Agg")

SEQUENTIAL_DATASET_NAMES = ["SimulationDatasetCutsSequential", "DataPointSequence", "Subset"]

FONT_SIZES = {
    "title": 13,
    "axis_label": 13,
    "tick_label": 13,
    "colorbar_label": 13,
    "colorbar_tick": 13,
    "suptitle": 13,
}

SAVEFIG_DEFAULTS = {
    "bbox_inches": "tight",
    "pad_inches": 0.02,
}


@dataclass
class DataToVisualize:
    datasetType: DatasetType
    data: np.ndarray
    category: str
    physical_property: str
    extent_highs: tuple[float, float] = (1280, 100)
    imshowargs: dict = field(default_factory=dict)
    vmax: float | None = None
    vmin: float | None = None
    dark_mode: bool = False
    temperature_spread = 5
    ambient_temp = 10.6

    def __post_init__(self):
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.detach().cpu().numpy()
        if not isinstance(self.extent_highs, tuple):
            self.extent_highs = tuple(float(v) for v in self.extent_highs)

        extent = (0, int(self.extent_highs[0]), int(self.extent_highs[1]), 0)

        match self.physical_property:
            case (
                "Liquid X-Velocity [m_per_y]"
                | "Liquid Y-Velocity [m_per_y]"
                | "Liquid Z-Velocity [m_per_y]"
                | "Liquid Pressure [Pa]"
                | "Permeability X [m^2]"
                | "Pressure Gradient [-]"
                | "SDF"
                | "Absolute Times"
                | "Gap times"
            ):
                cmap = "jp_linear_dark" if self.dark_mode else "jp_linear"
            case (
                "Streamlines Faded [-]"
                | "Streamlines Faded Outer [-]"
                | "Streamline-Sum_Position"
                | "Streamline-Sum_RelativeUncertainty"
                | "Streamline-Sum_TimeFaded-Position"
                | "Streamline-Max_TimeFaded"
                | "Streamline-Max_TimeSeasons"
            ):
                self.vmin = 0
                self.vmax = 1
                cmap = "jp_linear_dark" if self.dark_mode else "jp_linear"
            case "Material ID":
                cmap = "binary"
                self.vmin = 0
                self.vmax = 1
            case "Line Integral Convolution" | "LIC":
                cmap = "bone"
            case "Temperature [C]" | "Streamline-TemperatureApproximation" | "Streamline-Sum_TimeSeasons-Position":
                if self.category == "Absolute Error":
                    cmap = "jp_linear_dark" if self.dark_mode else "jp_linear"
                elif self.physical_property in [
                    "Streamline-TemperatureApproximation",
                    "Streamline-Sum_TimeSeasons-Position",
                ]:
                    cmap = "jp_linear_dark" if self.dark_mode else "jp_linear"
                    self.vmin = 0
                    self.vmax = 1
                else:
                    match self.datasetType:
                        case DatasetType.seasonal:
                            cmap = "jp_temperature_bidirectional"
                            self.vmin = self.ambient_temp - self.temperature_spread
                            self.vmax = self.ambient_temp + self.temperature_spread
                        case DatasetType.steady_state_heating:
                            cmap = "jp_temperature_upperlinear"
                            self.vmin = self.ambient_temp
                            self.vmax = self.ambient_temp + self.temperature_spread
                        case DatasetType.steady_state_cooling:
                            cmap = "jp_temperature_lowerlinear"
                            self.vmin = self.ambient_temp - self.temperature_spread
                            self.vmax = self.ambient_temp
                        case _:
                            raise ValueError(f"Unknown dataset type: {self.datasetType}")
                    if self.dark_mode:
                        cmap += "_dark"
            case _:
                raise ValueError(f"Unknown physical property: {self.physical_property}")

        self.imshowargs = {
            "cmap": cmap,
            "extent": extent,
            "interpolation": "nearest",
        }

        if self.vmax is not None:
            self.imshowargs["vmax"] = self.vmax
        if self.vmin is not None:
            self.imshowargs["vmin"] = self.vmin

        self._normalize_labels()

    def _normalize_labels(self):
        mapping = {
            "Liquid Pressure [Pa]": "Pressure [Pa]",
            "Material ID": "Positions of Heat Pumps [-]",
            "Permeability X [m^2]": "Permeability [m$^2$]",
            "SDF": "SDF-Transformed Positions of Heat Pumps [-]",
            "MDF": "MDF-Transformed Positions of Heat Pumps [-]",
            "Streamlines Fade": "Streamlines Fade [-]",
            "Streamlines": "Streamlines [-]",
        }
        if self.physical_property in mapping:
            self.physical_property = mapping[self.physical_property]


def _unpack_batch(batch):
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


def _get_dataset_context(dataloader):
    try:
        norm = dataloader.dataset.norm
        info = dataloader.dataset.info
        dataset = dataloader.dataset
    except AttributeError:
        norm = dataloader.dataset.dataset.norm
        info = dataloader.dataset.dataset.info
        dataset = dataloader.dataset.dataset
    return norm, info, dataset


def _is_sequential_dataset(dataset) -> bool:
    return dataset.__class__.__name__ in SEQUENTIAL_DATASET_NAMES


def aligned_colorbar(
    ax,
    im,
    colorbar_label_size: int = FONT_SIZES["colorbar_label"],
    colorbar_tick_size: int = FONT_SIZES["colorbar_tick"],
    **kwargs,
):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.3, pad=0.05)
    cbar = ax.figure.colorbar(im, cax=cax, **kwargs)
    cbar.ax.tick_params(labelsize=colorbar_tick_size)
    if kwargs.get("label"):
        cbar.set_label(kwargs["label"], fontsize=colorbar_label_size)
    return cbar


def plot_datapoint(
    name: str,
    datapoint: DataToVisualize,
    name_pic: str,
    settings_pic: dict,
    only_inner: bool = False,
    remove_axis: bool = False,
    dark_mode: bool = False,
):
    is_streamline = name.startswith("Streamline")

    if remove_axis and is_streamline:
        fig = Figure(figsize=(6, 5))
        settings = settings_pic.copy()
        settings["dpi"] = 2560 / 5
        settings["bbox_inches"] = "tight"
        settings["pad_inches"] = 0
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
    else:
        fig = Figure(figsize=(6.4, 5))
        settings = settings_pic.copy()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(datapoint.category, fontsize=FONT_SIZES["title"])

    imshow_args = datapoint.imshowargs.copy()

    if "error" in name_pic.lower():
        imshow_args["cmap"] = "jp_linear"
        imshow_args.pop("vmax", None)
        imshow_args.pop("vmin", None)

    if dark_mode and is_streamline:
        imshow_args["cmap"] += "_dark"

    data_to_show = datapoint.data[100:400, 100:400] if only_inner else datapoint.data
    im = ax.imshow(data_to_show, **imshow_args)
    ax.invert_yaxis()

    if datapoint.vmax is not None and datapoint.vmin is not None and "error" not in name_pic.lower():
        im.set_clim(datapoint.vmin, datapoint.vmax)

    if not (remove_axis and is_streamline):
        ax.set_ylabel("x [m]", fontsize=FONT_SIZES["axis_label"])
        ax.set_xlabel("y [m]", fontsize=FONT_SIZES["axis_label"])
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["tick_label"])
        aligned_colorbar(ax, im, label=datapoint.physical_property)
        fig.set_tight_layout(True)

    ext_inner = "_inner" if only_inner else ""
    save_settings = settings.copy()
    for key, value in SAVEFIG_DEFAULTS.items():
        save_settings.setdefault(key, value)
    fig.savefig(f"{name_pic}{ext_inner}.{settings['format']}", **save_settings)


def plot_datafields(
    data: dict[str, DataToVisualize],
    name_pic: str,
    settings_pic: dict,
    only_inner: bool = False,
    plot_all_in_1_pic: bool = True,
):
    if plot_all_in_1_pic:
        num_subplots = len(data)
        fig = Figure(figsize=(6.4, num_subplots * 3))

        axes = fig.subplots(num_subplots, 1, sharex=True)
        if num_subplots == 1:
            axes = [axes]

        for index, (_, datapoint) in enumerate(data.items()):
            ax = axes[index]
            ax.set_title(datapoint.category, fontsize=FONT_SIZES["title"])

            data_to_show = datapoint.data[100:400, 100:400] if only_inner else datapoint.data
            ax.imshow(data_to_show, **datapoint.imshowargs)

            ax.invert_yaxis()
            ax.set_ylabel("x [m]", fontsize=FONT_SIZES["axis_label"])
            ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["tick_label"])

        axes[-1].set_xlabel("y [m]", fontsize=FONT_SIZES["axis_label"])
        fig.set_tight_layout(True)

        ext_inner = "_inner" if only_inner else ""
        save_settings = settings_pic.copy()
        for key, value in SAVEFIG_DEFAULTS.items():
            save_settings.setdefault(key, value)
        fig.savefig(f"{name_pic}{ext_inner}.{settings_pic['format']}", **save_settings)
    else:
        for name, datapoint in data.items():
            log.info(f"Plotting {name}...")
            plot_datapoint(name, datapoint, f"{name_pic}_{name}", settings_pic, only_inner)


def prepare_data_to_plot_sequential(
    datasetType: DatasetType, x: torch.Tensor, y: torch.Tensor, y_out: torch.Tensor, info: dict
):
    y_reduced = y.squeeze_()
    y_out = y_out.squeeze_()
    outs_max = [max(y_reduced.max(), y_out.max()) for _ in range(len(y_reduced))]
    outs_min = [min(y_reduced.min(), y_out.min()) for _ in range(len(y_reduced))]
    extent_highs_y = tuple(np.array(info["CellsSize"][:2]) * y_out.shape[-2:])

    dict_to_plot = {}
    labels = info["Labels"].keys()

    for label in labels:
        index = info["Labels"][label]["index"]
        for time_step in range(y_reduced.shape[0]):
            dict_to_plot[f"{label}_true at time {time_step}"] = DataToVisualize(
                datasetType,
                y_reduced[time_step],
                "Label",
                label,
                extent_highs_y,
                vmax=outs_max[index],
                vmin=outs_min[index],
            )
            dict_to_plot[f"{label}_out at time {time_step}"] = DataToVisualize(
                datasetType,
                y_out[time_step],
                "Prediction",
                label,
                extent_highs_y,
                vmax=outs_max[index],
                vmin=outs_min[index],
            )
            dict_to_plot[f"{label}_error at time {time_step}"] = DataToVisualize(
                datasetType,
                torch.abs(y_reduced[time_step] - y_out[time_step]),
                "Absolute Error",
                label,
                extent_highs_y,
            )

    for input_name in info["Inputs"].keys():
        index = info["Inputs"][input_name]["index"]
        extent_vals = tuple(np.array(info["CellsSize"][:2]) * x.shape[-2:])
        dict_to_plot[input_name] = DataToVisualize(
            datasetType, x[index].squeeze_(), "Input", input_name, extent_vals
        )

    return dict_to_plot


def prepare_data_to_plot_sequential_outputs(
    datasetType: DatasetType,
    y: torch.Tensor,
    y_out: torch.Tensor,
    plot_true: bool,
    info: dict,
    tiled: bool,
):
    y_reduced = y.squeeze_(1)[..., 10:-10, 10:-10]
    y_out = y_out.squeeze_(1)[..., 10:-10, 10:-10]
    outs_max = [max(y_reduced.max(), y_out.max()) for _ in range(len(y_reduced))]
    outs_min = [min(y_reduced.min(), y_out.min()) for _ in range(len(y_reduced))]
    extent_highs_y = tuple(np.array(info["CellsSize"][:2]) * y_out.shape[-2:])

    dict_to_plot = {}
    labels = info["Labels"].keys()
    tiled_str = " tiled" if tiled else ""

    if y_reduced.dim() > 3:
        y_reduced = y_reduced.squeeze_(0)
    if y_out.dim() > 3:
        y_out = y_out.squeeze_(0)

    for label in labels:
        index = info["Labels"][label]["index"]
        for time_step in range(y_reduced.shape[0]):
            assert len(y_reduced[time_step].shape) == 2 and len(y_out[time_step].shape) == 2, (
                f"Shape are not 2D: {y_reduced[time_step].shape}, {y_out[time_step].shape}"
            )
            if plot_true:
                dict_to_plot[f"{label}_true_at_time_{time_step}{tiled_str}"] = DataToVisualize(
                    datasetType,
                    y_reduced[time_step],
                    "Label",
                    label,
                    extent_highs_y,
                    vmax=outs_max[index],
                    vmin=outs_min[index],
                )
            dict_to_plot[f"{label}_out_at_time_{time_step}{tiled_str}"] = DataToVisualize(
                datasetType,
                y_out[time_step],
                "Prediction",
                label,
                extent_highs_y,
                vmax=outs_max[index],
                vmin=outs_min[index],
            )
            dict_to_plot[f"{label}_error_at_time_{time_step}{tiled_str}"] = DataToVisualize(
                datasetType,
                torch.abs(y_reduced[time_step] - y_out[time_step]),
                "Absolute Error",
                label,
                extent_highs_y,
            )

    return dict_to_plot


def prepare_data_to_plot_sequential_inputs(
    datasetType: DatasetType, x: torch.Tensor, y: torch.Tensor, info: dict
):
    dict_to_plot = {}
    inputs = info["Inputs"].keys()
    extent_vals = tuple(np.array(info["CellsSize"][:2]) * x.shape[-2:])

    for input_name in inputs:
        index = info["Inputs"][input_name]["index"]
        dict_to_plot[input_name] = DataToVisualize(
            datasetType, x[index, 0].squeeze_(), "Input", input_name, extent_vals
        )

    for i, input_name in enumerate(["Absolute Times", "Gap times"]):
        for time in range(x.size(1)):
            dict_to_plot[f"{input_name}_t{time}"] = DataToVisualize(
                datasetType,
                x[i + len(inputs), time].squeeze_(),
                f"Time step {time}",
                input_name,
                extent_vals,
            )

    if "Temperature [C]" in info["Labels"]:
        temp_idx = info["Labels"]["Temperature [C]"]["index"]
        if y.dim() == 4:
            temp_data = y[temp_idx, 0]
        elif y.dim() == 3:
            temp_data = y[0]
        else:
            temp_data = y[temp_idx]
    else:
        temp_data = x[-1].squeeze_()
        if isinstance(temp_data, torch.Tensor) and temp_data.dim() == 3:
            temp_data = temp_data[0]

    dict_to_plot["Temperature"] = DataToVisualize(
        datasetType, temp_data, "Input", "Temperature [C]", extent_vals
    )

    return dict_to_plot


def plot_datafields_sequential(data: dict[str, DataToVisualize], name_pic: str, settings_pic: dict):
    """Groups entries by physical_property, plots one figure per property with horizontal timesteps."""
    from collections import defaultdict

    groups = defaultdict(dict)

    for key, datapoint in data.items():
        try:
            t = int(key.rsplit("_t", 1)[-1])
        except ValueError:
            t = 0
        groups[datapoint.physical_property][t] = datapoint

    for prop_name, timestep_dict in groups.items():
        timesteps = sorted(timestep_dict.keys())
        n_t = len(timesteps)

        fig = Figure(figsize=(4 * n_t, 4))
        axes = fig.subplots(1, n_t, sharey=True)
        if n_t == 1:
            axes = [axes]

        for ax, t in zip(axes, timesteps, strict=False):
            datapoint = timestep_dict[t]
            ax.set_title(datapoint.category, fontsize=FONT_SIZES["title"])
            im = ax.imshow(datapoint.data, **datapoint.imshowargs)
            ax.invert_yaxis()
            ax.set_xlabel("y [m]", fontsize=FONT_SIZES["axis_label"])
            if t == timesteps[0]:
                ax.set_ylabel("x [m]", fontsize=FONT_SIZES["axis_label"])
            ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["tick_label"])
            aligned_colorbar(ax, im, label=prop_name)

        fig.suptitle(prop_name, fontsize=FONT_SIZES["suptitle"])
        fig.set_tight_layout(True)

        safe_name = prop_name.replace("/", "_").replace(" ", "_").replace("[", "").replace("]", "")
        save_settings = settings_pic.copy()
        for key, value in SAVEFIG_DEFAULTS.items():
            save_settings.setdefault(key, value)
        fig.savefig(f"{name_pic}_{safe_name}.{settings_pic['format']}", **save_settings)


def reverse_norm_one_dp_sequence(x: torch.Tensor, y: torch.Tensor, y_out: torch.Tensor, norm: NormalizeTransform):
    x = norm.reverse(x.detach().cpu(), "Inputs")
    y = norm.reverse(y.detach().cpu(), "Labels")
    try:
        y_out = y_out.detach().cpu().squeeze(0)
    except (AttributeError, TypeError):
        y_out = y_out.squeeze(0)
    y_out = norm.reverse(y_out, "Labels")
    return x, y, y_out


def reverse_norm_one_dp_outputs(y_out: torch.Tensor, norm: NormalizeTransform):
    y_sq = y_out.detach().cpu().squeeze(0)
    try:
        return norm.reverse(y_sq, "Labels")
    except TypeError:
        return norm.reverse(y_out.squeeze(0), "Labels")


def reverse_norm_one_dp_inputs(x: torch.Tensor, y: torch.Tensor, norm: NormalizeTransform):
    x_rev = norm.reverse(x.detach().cpu().squeeze(0), "Inputs")
    y_detached = y.detach().cpu()
    if len(y.shape) == 4:
        y_rev = norm.reverse(y_detached.squeeze(0), "Labels")
    else:
        y_rev = norm.reverse(y_detached, "Labels")
    return x_rev, y_rev


def prepare_data_to_plot_inputs(
    datasetType: DatasetType, x: torch.Tensor, y: torch.Tensor, info: dict
) -> dict[str, DataToVisualize]:
    required_size = y.shape
    h_diff = y.shape[1] - required_size[1]
    w_diff = y.shape[2] - required_size[2]

    start_h = h_diff // 2
    start_w = w_diff // 2
    y_reduced = y[:, start_h : start_h + required_size[1], start_w : start_w + required_size[2]]

    num_channels = len(y_reduced)
    outs_max = [y_reduced[i].max().item() for i in range(num_channels)]
    outs_min = [y_reduced[i].min().item() for i in range(num_channels)]

    dict_to_plot = {}
    for input_name, input_info in info["Inputs"].items():
        idx = input_info["index"]
        extent_vals = tuple(np.array(info["CellsSize"][:2]) * x.shape[-2:])
        dict_to_plot[input_name] = DataToVisualize(
            datasetType=datasetType,
            data=x[idx],
            category="",
            physical_property=input_name,
            extent_highs=extent_vals,
        )
    for label_name, label_info in info["Labels"].items():
        idx = label_info["index"]
        extent_vals = tuple(np.array(info["CellsSize"][:2]) * y.shape[-2:])
        dict_to_plot[f"{label_name}_true"] = DataToVisualize(
            datasetType=datasetType,
            data=y_reduced[idx],
            category="Label",
            physical_property=label_name,
            extent_highs=extent_vals,
            vmax=outs_max[idx],
            vmin=outs_min[idx],
        )

    return dict_to_plot


def prepare_data_to_plot_outputs(
    datasetType: DatasetType, y: torch.Tensor, y_out: torch.Tensor, info: dict, lic: bool = False
) -> dict[str, DataToVisualize]:
    required_size = y_out.shape
    h_diff = y.shape[1] - required_size[1]
    w_diff = y.shape[2] - required_size[2]

    start_h = h_diff // 2
    start_w = w_diff // 2
    y_reduced = y[:, start_h : start_h + required_size[1], start_w : start_w + required_size[2]]

    num_channels = len(y_reduced)
    outs_max = [y_reduced[i].max().item() for i in range(num_channels)]
    outs_min = [y_reduced[i].min().item() for i in range(num_channels)]

    extent_vals = tuple(np.array(info["CellsSize"][:2]) * y_out.shape[-2:])
    dict_to_plot = {}

    if lic:
        import lic

        index = info["Labels"]["Liquid X-Velocity [m_per_y]"]["index"]
        temp_x = y_reduced[index].cpu().numpy()
        index = info["Labels"]["Liquid Y-Velocity [m_per_y]"]["index"]
        temp_y = y_reduced[index].cpu().numpy()

        lic_result = lic.lic(temp_y, temp_x, length=30)
        dict_to_plot["LIC"] = DataToVisualize(
            datasetType,
            lic_result,
            "LIC",
            "Line Integral Convolution",
            extent_vals,
            vmax=np.max(lic_result),
            vmin=np.min(lic_result),
        )

    for label_name, label_info in info["Labels"].items():
        idx = label_info["index"]
        dict_to_plot[f"{label_name}_out"] = DataToVisualize(
            datasetType=datasetType,
            data=y_out[idx],
            category="Prediction",
            physical_property=label_name,
            extent_highs=extent_vals,
            vmax=outs_max[idx],
            vmin=outs_min[idx],
        )
        dict_to_plot[f"{label_name}_error"] = DataToVisualize(
            datasetType=datasetType,
            data=torch.abs(y_reduced[idx] - y_out[idx]),
            category="Absolute Error",
            physical_property=label_name,
            extent_highs=extent_vals,
        )

    return dict_to_plot


def plot_output_over_input(
    input_data: dict[str, DataToVisualize], output_data: dict[str, DataToVisualize], name_pic: str, settings_pic: dict
):
    """Plot output data overlaid on top of input data with transparency."""
    num_subplots = len(input_data)
    fig = Figure(figsize=(6.4, num_subplots * 3))

    axes = fig.subplots(num_subplots, 1, sharex=True)
    if num_subplots == 1:
        axes = [axes]

    for index, (name, input_datapoint) in enumerate(input_data.items()):
        imshow_args = input_datapoint.imshowargs.copy()

        ax = axes[index]
        ax.set_title(
            f"{input_datapoint.category} (Input: {input_datapoint.physical_property})",
            fontsize=FONT_SIZES["title"],
        )

        data_to_show_input = input_datapoint.data
        ax.imshow(data_to_show_input, **imshow_args, alpha=0.6)

        output_datapoint = None
        candidate_names = [name, name.replace("_true", "_out"), name.replace("_out", "_true")]
        for candidate in candidate_names:
            if candidate in output_data:
                output_datapoint = output_data[candidate]
                break

        if output_datapoint is None:
            for candidate in output_data.values():
                if candidate.physical_property == input_datapoint.physical_property:
                    output_datapoint = candidate
                    break

        if output_datapoint is None:
            for candidate in output_data.values():
                if candidate.physical_property == "Temperature [C]":
                    output_datapoint = candidate
                    break

        if output_datapoint is not None:
            data_to_show_output = output_datapoint.data
            output_imshow_args = output_datapoint.imshowargs.copy()
            ax.imshow(data_to_show_output, **output_imshow_args, alpha=0.4)

        ax.invert_yaxis()
        ax.set_ylabel("x [m]", fontsize=FONT_SIZES["axis_label"])
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["tick_label"])

    axes[-1].set_xlabel("y [m]", fontsize=FONT_SIZES["axis_label"])
    fig.set_tight_layout(True)
    save_settings = settings_pic.copy()
    for key, value in SAVEFIG_DEFAULTS.items():
        save_settings.setdefault(key, value)
    fig.savefig(f"{name_pic}.{settings_pic['format']}", **save_settings)


def visualize_inputs(
    datasetType: DatasetType,
    dataloader,
    args: dict,
    amount_datapoints_to_visu: int = inf,
    plot_path: str = "default",
    pic_format: str = "png",
):
    log.info("Visualizing Inputs...")

    if dataloader.dataset.__class__.__name__ == "SimulationDatasetCutsSequential":
        log.info("Skipping input visualization for SimulationDatasetCutsSequential training cutouts.")
        return

    norm, info, dataset = _get_dataset_context(dataloader)
    settings_pic = {"format": pic_format, "dpi": 160}
    is_sequential = _is_sequential_dataset(dataset)
    n_subsets = len(dataset.subsets) if (is_sequential and hasattr(dataset, "subsets")) else 1

    chain_buffer = {}
    plotted_count = 0

    for batch in dataloader:
        inputs, labels, metadata = _unpack_batch(batch)
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            if is_sequential and n_subsets > 1 and metadata is not None:
                chain_idx = metadata["chain_idx"][i].item()
                subset_idx = metadata["subset_idx"][i].item()
            else:
                chain_idx = plotted_count
                subset_idx = 0

            if chain_idx not in chain_buffer:
                chain_buffer[chain_idx] = {}
            chain_buffer[chain_idx][subset_idx] = (inputs[i], labels[i])

            if len(chain_buffer[chain_idx]) < n_subsets:
                continue

            if plotted_count >= amount_datapoints_to_visu:
                return

            combined_dict = {}
            for s_idx in sorted(chain_buffer[chain_idx].keys()):
                x_s, y_s = chain_buffer[chain_idx][s_idx]
                x_rev, y_rev = reverse_norm_one_dp_inputs(x_s, y_s, norm)

                if is_sequential:
                    sub_dict = prepare_data_to_plot_sequential_inputs(datasetType, x_rev, y_rev, info)
                    if n_subsets > 1:
                        sub_dict = {f"subset{s_idx}_{k}": v for k, v in sub_dict.items()}
                else:
                    sub_dict = prepare_data_to_plot_inputs(datasetType, x_rev, y_rev, info)

                combined_dict.update(sub_dict)

            name_pic = f"{plot_path}_{plotted_count}_input"
            plot_datafields(combined_dict, name_pic, settings_pic, only_inner=False, plot_all_in_1_pic=False)

            del chain_buffer[chain_idx]
            plotted_count += 1


def visualize_outputs(
    datasetType: DatasetType,
    model,
    dataloader,
    args: dict,
    amount_datapoints_to_visu: int = inf,
    plot_path: str = "default",
    pic_format: str = "png",
    scaleBounds=None,
    useNonLinearCmap: bool = None,
    plot_true: bool = True,
):
    log.info("Visualizing Outputs...")

    norm, info, dataset = _get_dataset_context(dataloader)
    settings_pic = {"format": pic_format, "dpi": 160}
    device = args["device"]
    is_sequential = _is_sequential_dataset(dataset)
    n_subsets = len(dataset.subsets) if (is_sequential and hasattr(dataset, "subsets")) else 1

    chain_buffer = {}
    plotted_count = 0

    for batch in dataloader:
        inputs, labels, metadata = _unpack_batch(batch)
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            if is_sequential and n_subsets > 1 and metadata is not None:
                chain_idx = metadata["chain_idx"][i].item()
                subset_idx = metadata["subset_idx"][i].item()
            else:
                chain_idx = plotted_count
                subset_idx = 0

            if chain_idx not in chain_buffer:
                chain_buffer[chain_idx] = {}
            chain_buffer[chain_idx][subset_idx] = (inputs[i], labels[i])

            if len(chain_buffer[chain_idx]) < n_subsets:
                continue

            if plotted_count >= amount_datapoints_to_visu:
                return

            combined_dict = {}
            init_frame = None
            for s_idx in sorted(chain_buffer[chain_idx].keys()):
                x_s, y_s = chain_buffer[chain_idx][s_idx]
                tiled = x_s.shape[-1] > 500 or x_s.shape[-2] > 500

                if is_sequential:
                    if tiled and hasattr(model, "infer_tiled"):
                        y_out_raw = model.infer_tiled(x_s.unsqueeze(0), device)
                    else:
                        y_out_raw = model.infer(x_s.unsqueeze(0), device, init_frame=init_frame)

                    if hasattr(y_out_raw, "shape") and y_out_raw.ndim >= 3:
                        init_frame = y_out_raw[:, :, -1]

                    x_rev, y_rev, y_out_rev = reverse_norm_one_dp_sequence(x_s, y_s, y_out_raw, norm)
                    sub_dict = prepare_data_to_plot_sequential_outputs(
                        datasetType, y_rev, y_out_rev, plot_true, info, tiled
                    )
                else:
                    y_out_raw = model.infer(x_s.unsqueeze(0), device)
                    y_out_rev = reverse_norm_one_dp_outputs(y_out_raw, norm)
                    _, y_rev = reverse_norm_one_dp_inputs(x_s, y_s, norm)
                    sub_dict = prepare_data_to_plot_outputs(datasetType, y_rev, y_out_rev, info)

                if n_subsets > 1:
                    sub_dict = {f"subset{s_idx}_{k}": v for k, v in sub_dict.items()}
                combined_dict.update(sub_dict)

            name_pic = f"{plot_path}_{plotted_count}_output"
            plot_datafields(combined_dict, name_pic, settings_pic, only_inner=False, plot_all_in_1_pic=False)

            del chain_buffer[chain_idx]
            plotted_count += 1


def visualize_outputs_over_inputs(
    datasetType: DatasetType,
    model: UNet,
    dataloader,
    args: dict,
    amount_datapoints_to_visu: int = inf,
    plot_path: str = "default",
    pic_format: str = "png",
):
    log.info("Visualizing Outputs over Inputs...")
    limit = min(amount_datapoints_to_visu, len(dataloader.dataset))

    norm, info, dataset = _get_dataset_context(dataloader)
    settings_pic = {"format": pic_format, "dpi": 160}
    device = args["device"]
    is_sequential = _is_sequential_dataset(dataset)

    current_count = 0

    for batch in dataloader:
        inputs, labels, _metadata = _unpack_batch(batch)
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            if current_count >= limit:
                return

            threshold_check = _label_has_midcell_above_threshold(labels[i], 0.5)
            is_val = "val" in str(plot_path)
            if not threshold_check and not is_val:
                continue

            x = inputs[i]
            y = labels[i]

            if is_sequential:
                y_out = model.infer(x.unsqueeze(0), device)
                x_t_denorm, y_denorm = reverse_norm_one_dp_inputs(x, y, norm)
                y_out_denorm = reverse_norm_one_dp_outputs(y_out, norm)

                input_dict = prepare_data_to_plot_sequential_inputs(datasetType, x_t_denorm, y_denorm, info)
                output_dict = prepare_data_to_plot_sequential_outputs(
                    datasetType, y_denorm, y_out_denorm, plot_true=True, info=info, tiled=False
                )

                temp_true_dict = {
                    k: v
                    for k, v in output_dict.items()
                    if v.physical_property == "Temperature [C]" and v.category == "Label"
                }
                if temp_true_dict:
                    output_dict = temp_true_dict
                else:
                    output_dict = {k: v for k, v in output_dict.items() if v.physical_property == "Temperature [C]"}

                name_pic = f"{plot_path}_{current_count}_output_over_input"
                plot_output_over_input(input_dict, output_dict, name_pic, settings_pic)
            else:
                x_copy = deepcopy(x)
                y_copy = deepcopy(y)

                y_out_raw = model.infer(x_copy.unsqueeze(0), device)
                y_out = reverse_norm_one_dp_outputs(y_out_raw, norm)
                x_denorm, y_denorm = reverse_norm_one_dp_inputs(x_copy, y_copy, norm)

                input_dict = prepare_data_to_plot_inputs(datasetType, x_denorm, y_denorm, info)
                output_dict = prepare_data_to_plot_outputs(datasetType, y_denorm, y_out, info, lic=False)
                temp_true_dict = {
                    k: v
                    for k, v in output_dict.items()
                    if v.physical_property == "Temperature [C]" and v.category == "Label"
                }
                if temp_true_dict:
                    output_dict = temp_true_dict
                else:
                    output_dict = {k: v for k, v in output_dict.items() if v.physical_property == "Temperature [C]"}

                name_pic = f"{plot_path}_{current_count}_output_over_input"
                plot_output_over_input(input_dict, output_dict, name_pic, settings_pic)

            current_count += 1


def visualize_streamlines(
    datasetType: DatasetType, output_name: str, prop_name: str, tensor_data: torch.Tensor
) -> None:
    """Generates plots for all streamlines."""
    resolution = (12800, 12800)

    plot_datapoint(
        name=f"Streamlines - {prop_name}",
        datapoint=DataToVisualize(datasetType, tensor_data, "Input", prop_name, resolution),
        name_pic=str(output_name),
        settings_pic={"format": "png", "dpi": 160},
        only_inner=False,
        remove_axis=False,
    )
