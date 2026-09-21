from code.postprocessing.cmap_jp import register_jp_temperature_cmaps  # noqa: F401
from code.postprocessing.cmap_jp import *  # noqa: F403
from code.preprocessing.datasets.dataset import DatasetType
from code.preprocessing.transforms import NormalizeTransform
from code.utils import logging as log  # noqa: F401
from copy import deepcopy
from dataclasses import dataclass, field
from math import inf
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.nn import Module
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import maximum_filter

matplotlib.use("Agg")


def extent_highs_for_plot(cells_size, spatial_shape) -> tuple[float, float]:
    """Map grid ``(nx, ny)`` lengths to imshow extents ``(y_max, x_max)``.

    Pipeline tensors are ``(..., nx, ny)`` with X vertical / Y horizontal in figures
    (``ylabel='x'``, ``xlabel='y'``). ``CellsSize[:2] * shape[-2:]`` yields ``(Lx, Ly)``;
    plots need ``(Ly, Lx)`` so the long plume axis is not drawn on the short scale.
    """
    lx_ly = np.asarray(cells_size[:2], dtype=float) * np.asarray(spatial_shape[-2:], dtype=float)
    return float(lx_ly[1]), float(lx_ly[0])


def prepare_field_for_display(datapoint: "DataToVisualize", data) -> np.ndarray:
    """Convert plot data to ndarray; dilate Material ID so single-cell HPs stay visible."""
    arr = np.asarray(data.detach().cpu() if torch.is_tensor(data) else data, dtype=np.float64)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    # Label may already be remapped by DataToVisualize._normalize_labels.
    if datapoint.physical_property not in {
        "Material ID",
        "Positions of Heat Pumps [-]",
        "SDF-Transformed Positions of Heat Pumps [-]",
        "MDF-Transformed Positions of Heat Pumps [-]",
    }:
        return arr
    # One cell on a ~1280×320 figure is sub-pixel; expand markers for display only.
    radius = max(3, min(arr.shape) // 80)
    return maximum_filter(arr, size=2 * radius + 1)


@dataclass
class DataToVisualize:
    datasetType: DatasetType
    data: np.ndarray
    category: str
    physical_property: str
    # (y_extent [m], x_extent [m]) — horizontal then vertical, matching xlabel/ylabel
    extent_highs: tuple[float, float] = (1280, 100)
    imshowargs: dict = field(default_factory=dict)
    vmax: float | None = None
    vmin: float | None = None
    dark_mode: bool = False
    # From step2 physical_parameters (required; no hardcoded fallbacks)
    ambient_temperature_C: float = field(kw_only=True)
    temperature_spread_C: float = field(kw_only=True)

    def __post_init__(self):
        register_jp_temperature_cmaps(self.ambient_temperature_C, self.temperature_spread_C)
        extent = (0, int(self.extent_highs[0]), int(self.extent_highs[1]), 0)

        match self.physical_property:
            case (
                "Liquid X-Velocity [m_per_y]"
                | "Liquid X-Velocity NoHP [m_per_y]"
                | "Liquid Y-Velocity [m_per_y]"
                | "Liquid Y-Velocity NoHP [m_per_y]"
                | "Liquid Z-Velocity [m_per_y]"
                | "Liquid Pressure [Pa]"
                | "Permeability X [m^2]"
                | "Pressure Gradient [-]"
            ):
                cmap = "jp_linear_dark" if self.dark_mode else "jp_linear"
            case "Streamlines Faded [-]" | "Streamlines Faded Outer [-]":
                self.vmin = 0
                self.vmax = 1
                cmap = "jp_linear_dark" if self.dark_mode else "jp_linear"
            case "Material ID":
                cmap = "binary"
                self.vmin = 0
                self.vmax = 1
            case "Line Integral Convolution":
                cmap = "bone"
            case "Temperature [C]" | "Streamline-TemperatureApproximation":
                match self.datasetType:
                    case DatasetType.seasonal:
                        cmap = "jp_temperature_bidirectional"
                        self.vmin = self.ambient_temperature_C - self.temperature_spread_C
                        self.vmax = self.ambient_temperature_C + self.temperature_spread_C
                    case DatasetType.steady_state_heating:
                        cmap = "jp_temperature_upperlinear"
                        self.vmin = self.ambient_temperature_C
                        self.vmax = self.ambient_temperature_C + self.temperature_spread_C
                    case DatasetType.steady_state_cooling:
                        cmap = "jp_temperature_lowerlinear"
                        self.vmin = self.ambient_temperature_C - self.temperature_spread_C
                        self.vmax = self.ambient_temperature_C
                    case _:
                        raise ValueError(f"Unknown dataset type: {self.datasetType}")
                if self.dark_mode:
                    cmap += "_dark"
                if self.physical_property == "Streamline-TemperatureApproximation":
                    self.vmin = 0
                    self.vmax = 1
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


def aligned_colorbar(ax, im, **kwargs):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.3, pad=0.05)
    cb = ax.figure.colorbar(im, cax=cax, **kwargs)
    cb.ax.tick_params(labelsize=20)


def plot_datapoint(
    name: str,
    datapoint: DataToVisualize,
    name_pic: str,
    settings_pic: dict,
    only_inner: bool = False,
    remove_axis: bool = False,
):
    is_streamline = name.startswith("Streamline")

    if remove_axis and is_streamline:
        fig = Figure(figsize=(8, 6.5))
        # Use a copy to avoid side effects on the settings dict
        settings = settings_pic.copy()
        settings["bbox_inches"] = "tight"
        settings["pad_inches"] = 0

        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
    else:
        fig = Figure(figsize=(8.4, 6.5))
        settings = settings_pic
        ax = fig.add_subplot(1, 1, 1)
        settings["bbox_inches"] = "tight"
        ax.tick_params(axis="both", which="major", labelsize=20)

    imshow_args = datapoint.imshowargs.copy()

    if "error" in name_pic.lower():
        imshow_args["cmap"] = "jp_linear"
        imshow_args["vmax"] = None
        imshow_args["vmin"] = None

    raw = datapoint.data[100:400, 100:400] if only_inner else datapoint.data
    data_to_show = prepare_field_for_display(datapoint, raw)
    im = ax.imshow(data_to_show, **imshow_args)

    if "error" not in name_pic.lower() and datapoint.vmax is not None and datapoint.vmin is not None:
        im.set_clim(datapoint.vmin, datapoint.vmax)

    if not (remove_axis and is_streamline):
        aligned_colorbar(ax, im)
        fig.set_tight_layout(True)

    ext_inner = "_inner" if only_inner else ""
    fig.savefig(f"{name_pic}{ext_inner}.{settings['format']}", **settings)


def plot_datafields(
    data: dict[str, DataToVisualize],
    name_pic: str,
    settings_pic: dict,
    only_inner: bool = False,
    plot_all_in_1_pic: bool = True,
):
    if plot_all_in_1_pic:
        num_subplots = len(data)
        fig = Figure(figsize=(8.4, num_subplots * 4))

        axes = fig.subplots(num_subplots, 1, sharex=True)
        if num_subplots == 1:
            axes = [axes]

        for index, (_, datapoint) in enumerate(data.items()):
            ax = axes[index]
            ax.set_title(datapoint.category, fontsize=16)

            raw = datapoint.data[100:400, 100:400] if only_inner else datapoint.data
            data_to_show = prepare_field_for_display(datapoint, raw)
            ax.imshow(data_to_show, **datapoint.imshowargs)

            ax.invert_yaxis()
            ax.set_ylabel("x [m]", fontsize=20)
            ax.tick_params(axis="both", which="major", labelsize=20)

        axes[-1].set_xlabel("y [m]", fontsize=20)
        fig.set_tight_layout(True)

        ext_inner = "_inner" if only_inner else ""
        fig.savefig(f"{name_pic}{ext_inner}.{settings_pic['format']}", **settings_pic)
    else:
        for name, datapoint in data.items():
            log.info(f"Plotting {name}...")
            plot_datapoint(name, datapoint, f"{name_pic}_{name}", settings_pic, only_inner)


def prepare_data_to_plot_sequential(
    datasetType: DatasetType,
    x: torch.Tensor,
    y: torch.Tensor,
    y_out: torch.Tensor,
    info: dict,
    *,
    ambient_temperature_C: float,
    temperature_spread_C: float,
):
    # prepare data of temperature true, temperature out, error, physical variables (inputs)
    required_size = y_out.shape[-2:]
    log.info(f"Required size: {required_size}")
    # start_pos = ((y.shape[1] - required_size[1])//2, (y.shape[2] - required_size[2])//2)
    # y_reduced = y[:,start_pos[0]:start_pos[0]+required_size[1], start_pos[1]:start_pos[1]+required_size[2]]
    y_reduced = y.squeeze_()
    y_out = y_out.squeeze_()
    outs_max = [max(y_reduced.max(), y_out.max()) for idx in range(len(y_reduced))]
    outs_min = [min(y_reduced.min(), y_out.min()) for idx in range(len(y_reduced))]
    extent_highs_y = extent_highs_for_plot(info["CellsSize"], y_out.shape)
    temp_kw = {
        "ambient_temperature_C": ambient_temperature_C,
        "temperature_spread_C": temperature_spread_C,
    }

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
                **temp_kw,
            )
            dict_to_plot[f"{label}_out at time {time_step}"] = DataToVisualize(
                datasetType,
                y_out[time_step],
                "Prediction",
                label,
                extent_highs_y,
                vmax=outs_max[index],
                vmin=outs_min[index],
                **temp_kw,
            )
            dict_to_plot[f"{label}_error at time {time_step}"] = DataToVisualize(
                datasetType,
                torch.abs(y_reduced[time_step] - y_out[time_step]),
                "Absolute Error",
                label,
                extent_highs_y,
                **temp_kw,
            )
    inputs = info["Inputs"].keys()
    for input in inputs:
        index = info["Inputs"][input]["index"]
        dict_to_plot[input] = DataToVisualize(
            datasetType,
            x[index].squeeze_(),
            "Input",
            input,
            extent_highs_for_plot(info["CellsSize"], x.shape),
            **temp_kw,
        )

    return dict_to_plot


def reverse_norm_one_dp_sequence(x: torch.Tensor, y: torch.Tensor, y_out: torch.Tensor, norm: NormalizeTransform):
    # reverse transform for plotting real values
    x = norm.reverse(x.detach().cpu(), "Inputs")
    if len(y.shape) == 4:
        y = norm.reverse(y.detach().cpu(), "Labels")
    else:
        y = norm.reverse(y.detach().cpu(), "Labels")
    try:
        y_out = norm.reverse(y_out.detach().cpu().squeeze(0), "Labels")
    except Exception:
        y_out = norm.reverse(y_out.squeeze(0), "Labels")
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
    datasetType: DatasetType,
    x: torch.Tensor,
    y: torch.Tensor,
    info: dict,
    *,
    ambient_temperature_C: float,
    temperature_spread_C: float,
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
    temp_kw = {
        "ambient_temperature_C": ambient_temperature_C,
        "temperature_spread_C": temperature_spread_C,
    }

    dict_to_plot = {}
    for input_name, input_info in info["Inputs"].items():
        idx = input_info["index"]
        dict_to_plot[input_name] = DataToVisualize(
            datasetType=datasetType,
            data=x[idx],
            category="",
            physical_property=input_name,
            extent_highs=extent_highs_for_plot(info["CellsSize"], x.shape),
            **temp_kw,
        )
    for label_name, label_info in info["Labels"].items():
        idx = label_info["index"]
        dict_to_plot[f"{label_name}_true"] = DataToVisualize(
            datasetType=datasetType,
            data=y_reduced[idx],
            category="Label",
            physical_property=label_name,
            extent_highs=extent_highs_for_plot(info["CellsSize"], y.shape),
            vmax=outs_max[idx],
            vmin=outs_min[idx],
            **temp_kw,
        )

    return dict_to_plot


def prepare_data_to_plot_outputs(
    datasetType: DatasetType,
    y: torch.Tensor,
    y_out: torch.Tensor,
    info: dict,
    lic: bool = False,
    *,
    ambient_temperature_C: float,
    temperature_spread_C: float,
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

    extent_vals = extent_highs_for_plot(info["CellsSize"], y_out.shape)
    temp_kw = {
        "ambient_temperature_C": ambient_temperature_C,
        "temperature_spread_C": temperature_spread_C,
    }
    dict_to_plot = {}

    if lic:
        import lic

        index = info["Labels"]["Liquid X-Velocity [m_per_y]"]["index"]
        temp_x = y_reduced[index].cpu().numpy()  # Auf CPU/NumPy konvertieren
        index = info["Labels"]["Liquid Y-Velocity [m_per_y]"]["index"]
        temp_y = y_reduced[index].cpu().numpy()  # Auf CPU/NumPy konvertieren

        lic_result = lic.lic(temp_y, temp_x, length=30)
        dict_to_plot["LIC"] = DataToVisualize(
            datasetType,
            lic_result,
            "LIC",
            "Line Integral Convolution",
            extent_vals,
            vmax=np.max(lic_result),
            vmin=np.min(lic_result),
            **temp_kw,
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
            **temp_kw,
        )
        dict_to_plot[f"{label_name}_error"] = DataToVisualize(
            datasetType=datasetType,
            data=torch.abs(y_reduced[idx] - y_out[idx]),
            category="Absolute Error",
            physical_property=label_name,
            extent_highs=extent_vals,
            **temp_kw,
        )

    return dict_to_plot


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
        log.info("skipping ...")
        return

    total_samples = len(dataloader.dataset)
    limit = min(amount_datapoints_to_visu, total_samples)

    try:
        norm = dataloader.dataset.norm
        info = dataloader.dataset.info
    except AttributeError:
        norm = dataloader.dataset.dataset.norm
        info = dataloader.dataset.dataset.info

    settings_pic = {"format": pic_format, "dpi": 160}
    current_count = 0
    temp_kw = {
        "ambient_temperature_C": args["ambient_temperature_C"],
        "temperature_spread_C": args["temperature_spread_C"],
    }

    for inputs, labels in dataloader:
        log.info(inputs.shape, labels.shape, "shape of inputs and labels")
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            if current_count >= limit:
                return

            name_pic = f"{plot_path}_{current_count}_input"
            x, y = reverse_norm_one_dp_inputs(inputs[i], labels[i], norm)

            dict_to_plot = prepare_data_to_plot_inputs(datasetType, x, y, info, **temp_kw)
            plot_datafields(dict_to_plot, name_pic, settings_pic, only_inner=False, plot_all_in_1_pic=False)

            current_count += 1


def visualize_outputs(
    datasetType: DatasetType,
    model: Module,
    dataloader,
    args: dict,
    amount_datapoints_to_visu: int = inf,
    plot_path: str = "default",
    pic_format: str = "png",
    scaleBounds=None,
    useNonLinearCmap: bool = None,
):
    log.info("Visualizing Outputs...")
    total_samples = len(dataloader.dataset)
    limit = min(amount_datapoints_to_visu, total_samples)

    try:
        norm = dataloader.dataset.norm
        info = dataloader.dataset.info
    except AttributeError:
        norm = dataloader.dataset.dataset.norm
        info = dataloader.dataset.dataset.info

    settings_pic = {"format": pic_format, "dpi": 160}
    current_count = 0
    device = args["device"]
    temp_kw = {
        "ambient_temperature_C": args["ambient_temperature_C"],
        "temperature_spread_C": args["temperature_spread_C"],
    }

    for inputs, labels in dataloader:
        log.info(inputs.shape, labels.shape, "shape of inputs and labels")
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            if current_count >= limit:
                return

            name_pic = f"{plot_path}_{current_count}_output"
            x = inputs[i]
            y = labels[i]
            if dataloader.dataset.__class__.__name__ == "SimulationDatasetCutsSequential":
                y_out = model.infer(x.unsqueeze(0), args["device"])

                x = x[:-1]
                x, y, y_out = reverse_norm_one_dp_sequence(x, y, y_out, norm)
                dict_to_plot = prepare_data_to_plot_sequential(datasetType, x, y, y_out, info, **temp_kw)
            else:
                # deepcopy to avoid in-place operations messing up gradients
                x_copy = deepcopy(x)
                y_copy = deepcopy(y)

                y_out_raw = model.infer(x_copy.unsqueeze(0), device)

                y_out = reverse_norm_one_dp_outputs(y_out_raw, norm)
                _, y_denorm = reverse_norm_one_dp_inputs(x_copy, y_copy, norm)

                dict_to_plot = prepare_data_to_plot_outputs(
                    datasetType, y_denorm, y_out, info, lic=False, **temp_kw
                )

            plot_datafields(dict_to_plot, name_pic, settings_pic, only_inner=False, plot_all_in_1_pic=False)

            current_count += 1


def visualize_streamlines(
    datasetType: DatasetType,
    output_name: str,
    prop_name: str,
    tensor_data: torch.Tensor,
    *,
    cells_size: list[float] | tuple[float, ...],
    ambient_temperature_C: float,
    temperature_spread_C: float,
) -> None:
    """Generates plots for all streamlines."""
    extent_highs = extent_highs_for_plot(cells_size, tensor_data.shape)

    plot_datapoint(
        name=f"Streamlines - {prop_name}",
        datapoint=DataToVisualize(
            datasetType,
            tensor_data,
            "Input",
            prop_name,
            extent_highs,
            ambient_temperature_C=ambient_temperature_C,
            temperature_spread_C=temperature_spread_C,
        ),
        name_pic=str(output_name),
        settings_pic={"format": "png", "dpi": 160},
        only_inner=False,
        remove_axis=False,
    )


def _as_numpy_hw(data: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    """Detach/squeeze a field to a 2-D numpy array.

    A (1, 2, H, W) velocity stack is reduced to speed ``|q|`` so callers can pass the
    same tensor the RWPT kernel uses without pre-collapsing it.
    """
    if data is None:
        return None
    if torch.is_tensor(data):
        tensor = data.detach().cpu().float()
        if tensor.ndim == 4:
            tensor = torch.linalg.vector_norm(tensor[0], dim=0)
        array = tensor.numpy()
    else:
        array = np.asarray(data, dtype=np.float64)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D field after squeeze, got shape {array.shape}")
    return array.astype(np.float64, copy=False)


def _signed_overshoot_C(temp_C: np.ndarray, min_temp_C: float, max_temp_C: float) -> np.ndarray:
    """Positive = too hot, negative = too cold, zero inside the physical band."""
    return np.maximum(temp_C - max_temp_C, 0.0) - np.maximum(min_temp_C - temp_C, 0.0)


def _overlay_wells(ax, wells_rowcol_px: torch.Tensor | np.ndarray | None) -> None:
    if wells_rowcol_px is None:
        return
    wells = wells_rowcol_px.detach().cpu().numpy() if torch.is_tensor(wells_rowcol_px) else np.asarray(wells_rowcol_px)
    if wells.size == 0:
        return
    wells = np.atleast_2d(wells)
    ax.scatter(
        wells[:, 1],
        wells[:, 0],
        s=28,
        facecolors="none",
        edgecolors="k",
        linewidths=0.9,
        zorder=5,
    )


def visualize_rwpt_fields(
    out_path: str | Path,
    title: str,
    *,
    ambient_temp_C: float,
    min_temp_C: float,
    max_temp_C: float,
    temp_spread_C: float,
    resolution_m_per_px: float,
    temp_C: torch.Tensor | np.ndarray | None = None,
    acceptance: torch.Tensor | np.ndarray | None = None,
    acceptance_next: torch.Tensor | np.ndarray | None = None,
    velocity_m_per_year: torch.Tensor | np.ndarray | None = None,
    wells_rowcol_px: torch.Tensor | np.ndarray | None = None,
) -> None:
    """Multi-panel PNG of one RWPT multi-resolution stage (pixel coordinates).

    Drawn with ``interpolation='nearest'`` so a coarse 32×128 field stays visibly
    blocky instead of being bilinearly smoothed into looking like the fine grid.
    Wells are (row, col) in that level's own pixel units.
    """
    temp_hw = _as_numpy_hw(temp_C)
    acc_hw = _as_numpy_hw(acceptance)
    acc_next_hw = _as_numpy_hw(acceptance_next)
    speed_hw = _as_numpy_hw(velocity_m_per_year)
    overshoot_hw = None if temp_hw is None else _signed_overshoot_C(temp_hw, min_temp_C, max_temp_C)

    register_jp_temperature_cmaps(ambient_temp_C, temp_spread_C)
    if max_temp_C <= ambient_temp_C + 1e-9:
        temp_cmap = "jp_temperature_lowerlinear"
    elif min_temp_C >= ambient_temp_C - 1e-9:
        temp_cmap = "jp_temperature_upperlinear"
    else:
        temp_cmap = "jp_temperature_bidirectional"

    panels: list[tuple[np.ndarray, str, str, float | None, float | None]] = []
    if speed_hw is not None:
        panels.append((speed_hw, r"$|q|$ [m/y]", "jp_linear", 0.0, max(float(np.nanmax(speed_hw)), 1e-12)))
    if acc_hw is not None:
        acc_label = "Acceptance used [-]" if acc_next_hw is not None else "Acceptance [-]"
        panels.append((acc_hw, acc_label, "jp_linear", 0.0, 1.0))
    if acc_next_hw is not None:
        panels.append((acc_next_hw, "Acceptance after update [-]", "jp_linear", 0.0, 1.0))
    if temp_hw is not None:
        panels.append((temp_hw, "Temperature prior [C]", temp_cmap, min_temp_C, max_temp_C))
    if overshoot_hw is not None and float(np.nanmax(np.abs(overshoot_hw))) > 1e-9:
        peak = max(float(np.nanmax(np.abs(overshoot_hw))), 0.05)
        panels.append((overshoot_hw, "Signed overshoot [C]", "RdBu_r", -peak, peak))

    if not panels:
        return

    out_path = Path(out_path)
    if out_path.suffix.lower() != ".png":
        out_path = out_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_panels = len(panels)
    n_cols = 2 if n_panels > 2 else n_panels
    n_rows = int(np.ceil(n_panels / n_cols))
    fig = Figure(figsize=(5.8 * n_cols, 4.6 * n_rows + 0.4))
    axes = fig.subplots(n_rows, n_cols, squeeze=False)

    for idx, (field, label, cmap, vmin, vmax) in enumerate(panels):
        ax = axes[idx // n_cols][idx % n_cols]
        im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("column [px]")
        ax.set_ylabel("row [px]")
        ax.set_aspect("equal")
        _overlay_wells(ax, wells_rowcol_px)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=9)

    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(f"{title}\ndx = {resolution_m_per_px:.2f} m/px", fontsize=12)
    fig.set_layout_engine("tight")
    fig.savefig(out_path, format="png", dpi=140, bbox_inches="tight")
    log.info(f"RWPT viz wrote {out_path}")
