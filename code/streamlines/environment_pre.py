from code.postprocessing.visualization import visualize_streamlines
from code.preprocessing.datasets.dataset import DatasetType
from code.preprocessing.preparing_datasets.raw_data_loading import get_hp_location_raw_np
from code.preprocessing.preprocessing import expand_property_names
from code.preprocessing.transforms import NormalizeTransform
from code.processing.networks.unetVariants import UNetNoPad2
from code.utils import logging as log  # noqa: F401
from pathlib import Path

import numpy as np
import torch
from torch.nn import Module


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
