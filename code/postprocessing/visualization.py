"""Pipeline-facing visualization: turn tensors into plot specs (npy + yaml).

Each ``visualize_*`` always writes ``<stem>.npy`` + ``<stem>.yaml`` via ``code.plotting``
and, if ``render`` is true, also draws the image. Re-render later with
``python -m code.plotting <dir|yaml>``.
"""

from code.plotting.spec import (
    AxesSpec,
    ColormapSpec,
    DisplaySpec,
    FigureSpec,
    OutputSpec,
    OverlaySpec,
    PlotSpec,
    derived,
    panel_npy_name,
    write_figure,
    write_panel,
)
from code.preprocessing.datasets.dataset import DatasetType
from code.preprocessing.transforms import NormalizeTransform
from code.utils import logging as log  # noqa: F401
from copy import deepcopy
from dataclasses import dataclass, field
from math import inf
from pathlib import Path

import numpy as np
import torch
from torch.nn import Module

# Physical properties whose single-cell markers are dilated for display.
_MARKER_PROPERTIES = {
    "Material ID",
    "Positions of Heat Pumps [-]",
    "SDF-Transformed Positions of Heat Pumps [-]",
    "MDF-Transformed Positions of Heat Pumps [-]",
}


def as_numpy_hw(data: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
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


def extent_highs_for_plot(cells_size, spatial_shape) -> tuple[float, float]:
    """Map grid ``(nx, ny)`` lengths to imshow extents ``(y_max, x_max)``.

    Pipeline tensors are ``(..., nx, ny)`` with X vertical / Y horizontal in figures
    (``ylabel='x'``, ``xlabel='y'``). ``CellsSize[:2] * shape[-2:]`` yields ``(Lx, Ly)``;
    plots need ``(Ly, Lx)`` so the long plume axis is not drawn on the short scale.
    """
    lx_ly = np.asarray(cells_size[:2], dtype=float) * np.asarray(spatial_shape[-2:], dtype=float)
    return float(lx_ly[1]), float(lx_ly[0])


def signed_overshoot_C(temp_C: np.ndarray, min_temp_C: float, max_temp_C: float) -> np.ndarray:
    """Positive = too hot, negative = too cold, zero inside the physical band."""
    return np.maximum(temp_C - max_temp_C, 0.0) - np.maximum(min_temp_C - temp_C, 0.0)


def temperature_cmap_for_band(ambient_temp_C: float, min_temp_C: float, max_temp_C: float) -> str:
    if max_temp_C <= ambient_temp_C + 1e-9:
        return "jp_temperature_lowerlinear"
    if min_temp_C >= ambient_temp_C - 1e-9:
        return "jp_temperature_upperlinear"
    return "jp_temperature_bidirectional"


@dataclass
class DataToVisualize:
    datasetType: DatasetType
    data: np.ndarray
    category: str
    physical_property: str
    # (y_extent [m], x_extent [m]) — horizontal then vertical, matching xlabel/ylabel
    extent_highs: tuple[float, float] = (1280, 100)
    vmax: float | None = None
    vmin: float | None = None
    dark_mode: bool = False
    # From step2 physical_parameters (required; no hardcoded fallbacks)
    ambient_temperature_C: float = field(kw_only=True)
    temperature_spread_C: float = field(kw_only=True)
    cmap: str = field(init=False)

    def __post_init__(self):
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
        self.cmap = cmap
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

    def to_spec(self, name: str, pic_format: str = "png", dpi: float = 160) -> PlotSpec:
        """Panel spec; ``name`` containing "error" switches to free-scaled jp_linear."""
        cmap, vmin, vmax = self.cmap, self.vmin, self.vmax
        if "error" in name.lower():
            cmap, vmin, vmax = "jp_linear", None, None
        jp_params = None
        if cmap.startswith("jp_temperature"):
            jp_params = {
                "ambient_temperature_C": self.ambient_temperature_C,
                "temperature_spread_C": self.temperature_spread_C,
            }
        dilation = 0
        if self.physical_property in _MARKER_PROPERTIES:
            # One cell on a ~1280×320 figure is sub-pixel; expand markers for display only.
            dilation = max(3, min(as_numpy_hw(self.data).shape) // 80)
        return PlotSpec(
            title=self.category or None,
            quantity=self.physical_property,
            colormap=ColormapSpec(name=cmap, vmin=vmin, vmax=vmax, jp_params=jp_params),
            axes=AxesSpec(extent=[0, int(self.extent_highs[0]), int(self.extent_highs[1]), 0]),
            display=DisplaySpec(marker_dilation_px=dilation),
            output=OutputSpec(format=pic_format, dpi=dpi),
        )


def emit_panel(path_stem: str | Path, data, spec: PlotSpec, render: bool) -> Path:
    """Write npy+yaml for one panel and optionally render it; returns the yaml path."""
    yaml_path = write_panel(path_stem, as_numpy_hw(data), spec)
    if render:
        from code.plotting.render import render_panel

        render_panel(yaml_path)
    return yaml_path


def write_datafields(
    data: dict[str, DataToVisualize],
    name_pic: str,
    pic_format: str = "png",
    render: bool = True,
) -> None:
    """One npy+yaml (+ image) per field, named ``<name_pic>_<field name>``."""
    for name, datapoint in data.items():
        log.info(f"Plotting {name}...")
        emit_panel(f"{name_pic}_{name}", datapoint.data, datapoint.to_spec(name, pic_format), render)


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
    render: bool = True,
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
            write_datafields(dict_to_plot, name_pic, pic_format, render)

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
    render: bool = True,
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

            write_datafields(dict_to_plot, name_pic, pic_format, render)

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
    render: bool = True,
) -> None:
    """Writes the plot spec (and optionally the image) of one streamline field."""
    datapoint = DataToVisualize(
        datasetType,
        tensor_data,
        "Input",
        prop_name,
        extent_highs_for_plot(cells_size, tensor_data.shape),
        ambient_temperature_C=ambient_temperature_C,
        temperature_spread_C=temperature_spread_C,
    )
    emit_panel(output_name, tensor_data, datapoint.to_spec(f"Streamlines - {prop_name}"), render)


def write_prior_vs_true(
    out_stem: str | Path,
    prior_C: torch.Tensor | np.ndarray,
    true_C: torch.Tensor | np.ndarray,
    *,
    datasetType: DatasetType,
    cells_size: list[float] | tuple[float, ...],
    ambient_temperature_C: float,
    temperature_spread_C: float,
) -> tuple[list[Path], dict[str, float]]:
    """Three separate plot specs: thermal prior, ground truth and |prior − true| (all °C).

    Writes ``<stem>_prior.npy/.yaml``, ``<stem>_true.npy/.yaml`` and ``<stem>_error.yaml``
    (computed from the two npy on load, no npy of its own; MAE, RMSE and max error under
    ``meta``). Returns the three yaml paths and the metrics.
    """
    prior_hw = as_numpy_hw(prior_C)
    true_hw = as_numpy_hw(true_C)
    if prior_hw.shape != true_hw.shape:
        raise ValueError(f"Shape mismatch: prior {prior_hw.shape} vs true {true_hw.shape}")
    error_hw = np.abs(prior_hw - true_hw)
    metrics = {
        "mae_C": float(np.mean(error_hw)),
        "rmse_C": float(np.sqrt(np.mean(error_hw**2))),
        "max_abs_C": float(np.max(error_hw)),
    }

    extent_highs = extent_highs_for_plot(cells_size, prior_hw.shape)
    temp_kw = {"ambient_temperature_C": ambient_temperature_C, "temperature_spread_C": temperature_spread_C}
    yaml_paths = []
    for suffix, data, title in (
        ("prior", prior_hw, "Thermal prior [C]"),
        ("true", true_hw, "Ground truth [C]"),
        ("error", error_hw, "|prior - true| [C]"),
    ):
        spec = DataToVisualize(datasetType, data, title, "Temperature [C]", extent_highs, **temp_kw).to_spec(suffix)
        if suffix == "error":
            spec.colormap.vmin = 0.0
            spec.data = derived("abs_diff", panel_npy_name(out_stem, "prior"), panel_npy_name(out_stem, "true"))
            spec.meta = dict(metrics)
            data = None
        yaml_paths.append(write_panel(f"{out_stem}_{suffix}", data, spec))
    return yaml_paths, metrics


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
    render: bool = True,
) -> Path | None:
    """Multi-panel spec of one RWPT multi-resolution stage (pixel coordinates).

    Writes ``<stem>_<panel>.npy/.yaml`` per field plus ``<stem>.figure.yaml``; with
    ``render`` also ``<stem>.png``. Drawn with ``interpolation='nearest'`` so a coarse
    32×128 field stays visibly blocky instead of being bilinearly smoothed into looking
    like the fine grid. Wells are (row, col) in that level's own pixel units.
    """
    temp_hw = as_numpy_hw(temp_C)
    acc_hw = as_numpy_hw(acceptance)
    acc_next_hw = as_numpy_hw(acceptance_next)
    speed_hw = as_numpy_hw(velocity_m_per_year)
    overshoot_hw = None if temp_hw is None else signed_overshoot_C(temp_hw, min_temp_C, max_temp_C)
    temp_cmap = temperature_cmap_for_band(ambient_temp_C, min_temp_C, max_temp_C)

    # (file suffix, field, title, cmap, vmin, vmax)
    fields: list[tuple[str, np.ndarray, str, str, float | None, float | None]] = []
    if speed_hw is not None:
        fields.append(("speed", speed_hw, r"$|q|$ [m/y]", "jp_linear", 0.0, max(float(np.nanmax(speed_hw)), 1e-12)))
    if acc_hw is not None:
        acc_label = "Acceptance used [-]" if acc_next_hw is not None else "Acceptance [-]"
        fields.append(("acceptance", acc_hw, acc_label, "jp_linear", 0.0, 1.0))
    if acc_next_hw is not None:
        fields.append(("acceptance_next", acc_next_hw, "Acceptance after update [-]", "jp_linear", 0.0, 1.0))
    if temp_hw is not None:
        fields.append(("temperature", temp_hw, "Temperature prior [C]", temp_cmap, min_temp_C, max_temp_C))
    if overshoot_hw is not None and float(np.nanmax(np.abs(overshoot_hw))) > 1e-9:
        peak = max(float(np.nanmax(np.abs(overshoot_hw))), 0.05)
        fields.append(("overshoot", overshoot_hw, "Signed overshoot [C]", "RdBu_r", -peak, peak))

    if not fields:
        return None

    wells = [] if wells_rowcol_px is None else np.atleast_2d(
        wells_rowcol_px.detach().cpu().numpy() if torch.is_tensor(wells_rowcol_px) else np.asarray(wells_rowcol_px)
    ).tolist()
    jp_params = {"ambient_temperature_C": ambient_temp_C, "temperature_spread_C": temp_spread_C}

    panels = []
    for suffix, field_hw, label, cmap, vmin, vmax in fields:
        spec = PlotSpec(
            title=label,
            quantity=label,
            colormap=ColormapSpec(
                name=cmap,
                vmin=vmin,
                vmax=vmax,
                jp_params=jp_params if cmap.startswith("jp_temperature") else None,
                colorbar_labelsize=9,
                colorbar_size="4%",
            ),
            axes=AxesSpec(
                show_labels=True,
                show_title=True,
                xlabel="column [px]",
                ylabel="row [px]",
                tick_labelsize=10,
                title_fontsize=11,
            ),
            overlays=OverlaySpec(wells_rowcol_px=wells),
            output=OutputSpec(dpi=140, figsize=[5.8, 5.0]),
        )
        panels.append((suffix, field_hw, spec))

    out_path = Path(out_path)
    if out_path.suffix.lower() == ".png":
        out_path = out_path.parent / out_path.stem
    figure_path = write_figure(
        out_path,
        panels,
        FigureSpec(suptitle=f"{title}\ndx = {resolution_m_per_px:.2f} m/px", ncols=2),
    )
    if render:
        from code.plotting.render import render_figure

        render_figure(figure_path)
    log.info(f"RWPT viz wrote {figure_path}")
    return figure_path
