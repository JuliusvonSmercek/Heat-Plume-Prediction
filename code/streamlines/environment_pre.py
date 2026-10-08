from code.postprocessing.visualization import visualize_streamlines
from code.preprocessing.datasets.dataset import DatasetType
from code.preprocessing.preparing_datasets.raw_data_loading import get_hp_location_raw_np
from code.preprocessing.preprocessing import expand_property_names
from code.preprocessing.transforms import NormalizeTransform
from code.processing.networks.unetVariants import UNetNoPad2
from code.utils import logging as log  # noqa: F401
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module

# Fixed color scale for the absolute-difference panel [°C]
ABS_DIFF_VMIN = 0.0
ABS_DIFF_VMAX = 2.0


def compare_temperature_fields(
    a: np.ndarray,
    b: np.ndarray,
    *,
    title_a: str = "A",
    title_b: str = "B",
    out_path: Path,
    abs_diff_vmin: float = ABS_DIFF_VMIN,
    abs_diff_vmax: float = ABS_DIFF_VMAX,
) -> dict[str, float]:
    """Write a 1×3 figure (A, B, |A−B|) and return error metrics."""
    a = np.squeeze(np.asarray(a, dtype=np.float64))
    b = np.squeeze(np.asarray(b, dtype=np.float64))
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2-D fields, got shapes {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    abs_diff = np.abs(a - b)
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    max_abs = float(np.max(abs_diff))

    vmin = float(min(a.min(), b.min()))
    vmax = float(max(a.max(), b.max()))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    im0 = axes[0].imshow(a, origin="lower", vmin=vmin, vmax=vmax, cmap="coolwarm")
    axes[0].set_title(title_a)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(b, origin="lower", vmin=vmin, vmax=vmax, cmap="coolwarm")
    axes[1].set_title(title_b)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        abs_diff, origin="lower", vmin=abs_diff_vmin, cmap="magma"
    )
    axes[2].set_title(f"|{title_a} − {title_b}|")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=1600)
    plt.close(fig)

    metrics = {"mae": mae, "rmse": rmse, "max_abs": max_abs, "shape": a.shape}
    metrics_path = out_path.with_suffix(".txt")
    metrics_path.write_text(
        f"shape: {metrics['shape']}\n"
        f"MAE:     {metrics['mae']:.6g}\n"
        f"RMSE:    {metrics['rmse']:.6g}\n"
        f"max|Δ|:  {metrics['max_abs']:.6g}\n"
        f"wrote:   {out_path}\n",
        encoding="utf-8",
    )
    return metrics

def load_velocity_field(
    run_id: str,
    origin_path: Path,
    use_model: bool,
    model: Module | None,
    norm_transform: NormalizeTransform,
    device: torch.device,
) -> torch.Tensor:
    """Loads velocity from disk or infers it using the step1 velocity model."""
    if use_model and model is not None:
        data_in = torch.load(origin_path / "Inputs" / run_id)
        with torch.no_grad():
            velocity = model(data_in.unsqueeze(0).to(device)).cpu().detach().squeeze(0)
    else:
        velocity = torch.load(origin_path / "Labels" / run_id)

    norm_transform.reverse(velocity, "Labels")
    return velocity


def extract_heat_pump_positions(input_tensor: torch.Tensor, channel_index: int) -> np.ndarray:
    """Finds coordinates where Material ID == 1 (Heat Pumps). Returns shape (N, 2).

    ``get_hp_location_raw_np`` squeezes a single HP to shape (2,); reshape back to
    (2, 1) so step2 still gets one origin (multi-HP (2, N) is unchanged).
    """
    locs = np.asarray(get_hp_location_raw_np(input_tensor[channel_index].detach().cpu().numpy()))
    if locs.size == 0:
        return np.zeros((0, 2), dtype=float)
    if locs.ndim == 1:
        locs = locs.reshape(2, 1)
    return locs.T.astype(float) + 0.5


def save_result(
    destination_path: Path,
    run_id: str,
    inputs_tensor: torch.Tensor,
    streamlines: dict[str, torch.Tensor],
    norm_transform: NormalizeTransform,
    step3_in_map: dict[str, int],
    step2_in_map: dict[str, int],
) -> None:
    """Renormalizes and saves the final tensor to disk."""
    inputs_normed = norm_transform(inputs_tensor, "Inputs")

    # Inject calculated streamlines for keys present in step3 but missing in step2
    for key, idx in step3_in_map.items():
        if key not in step2_in_map and key in streamlines:
            inputs_normed[idx] = streamlines[key]

    output_dir = destination_path / "Inputs"
    output_dir.mkdir(exist_ok=True, parents=True)

    datapoint_path = output_dir / run_id
    torch.save(inputs_normed, datapoint_path)
    log.debug(f"Saved processed datapoint: {datapoint_path}")


def run_visualization(
    datasetType: DatasetType,
    streamlines: dict[str, torch.Tensor],
    results_path: Path,
    run_name: str,
    *,
    cells_size: list[float] | tuple[float, ...],
    ambient_temperature_C: float,
    temperature_spread_C: float,
) -> None:
    """Generates plots for all streamlines."""
    results_path.mkdir(exist_ok=True, parents=True)

    for key, tensor_data in streamlines.items():
        prop_name = expand_property_names(key)[0]
        output_name = results_path / f"{run_name}-{prop_name}"

        visualize_streamlines(
            datasetType,
            output_name,
            prop_name,
            tensor_data,
            cells_size=cells_size,
            ambient_temperature_C=ambient_temperature_C,
            temperature_spread_C=temperature_spread_C,
        )
