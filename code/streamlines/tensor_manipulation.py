import torch
from pathlib import Path
from shutil import copytree
from typing import Dict, List, Any

# --- Project Imports ---
from preprocessing.preprocessing import expand_property_names
from preprocessing.transforms import ToTensorTransform, get_transforms
import preprocessing.preparing_datasets.raw_data_loading as load
from utils.utils_args import load_yaml, save_yaml


def create_property_index_map(property_names: List[str]) -> Dict[str, int]:
    """Create a mapping from property name to their tensor indices."""
    return {name: i for i, name in enumerate(property_names)}


def expand_input_dimensions(
    inputs_normed: torch.Tensor,
    src_map: Dict[str, int],
    tgt_map: Dict[str, int]
) -> torch.Tensor:
    """Reshape input tensor to target config; maps existing channels and zeros new ones."""
    out = torch.zeros(
        (len(tgt_map), *inputs_normed.shape[1:]),
        dtype=inputs_normed.dtype,
        device=inputs_normed.device
    )
    for name, src_idx in src_map.items():
        if name in tgt_map:
            out[tgt_map[name]] = inputs_normed[src_idx]
    return out.float()
def prepare_dataset_step3(
    step2_dir: Path,
    dest_dir: Path,
    vel_model_dir: Path,
    use_vel_model: bool,
    dataset_name: str,
    step1_inputs: List[str],
    step1_outputs: List[str],
    step2_inputs: List[str],
    step3_inputs: List[str],
    step3_outputs: List[str]
) -> None:
    """Consolidate metadata, propagate normalization stats, and generate args.yaml."""
    copytree(step2_dir, dest_dir, dirs_exist_ok=True)

    # Load configuration files
    info = load_yaml(dest_dir / "info.yaml")
    #vel_info = load_yaml(vel_model_dir / "info.yaml")

    # Pre-load available statistics from Step 2 Inputs and Velocity Outputs
    stats_pool: Dict[str, Any] = {}

    # 1. Preserve stats from Step 2 (using Step 1 names for lookup compatibility)
    for key in step1_inputs:
        expanded = expand_property_names(key)[0]
        if expanded in info.get("Inputs", {}):
            stats_pool[key] = info["Inputs"][expanded]

    # 2. Capture stats from Velocity Model Outputs
    # for key in step1_outputs:
    #     expanded = expand_property_names(key)[0]
    #     if expanded in vel_info.get("Labels", {}):
    #         stats_pool[key] = vel_info["Labels"][expanded]

    # Rebuild Inputs section in destination info.yaml
    info["Inputs"] = {}
    step3_map = create_property_index_map(step3_inputs)
    
    # Collect expanded names for args.yaml
    expanded_step3_inputs = []

    for idx, key in enumerate(step3_inputs):
        props = expand_property_names(key)
        expanded_step3_inputs.extend(props)
        prop_name = props[0]  # Use primary name for stats

        # Assign stats: Use pool if available, else default synthetic values
        if key in stats_pool:
            info["Inputs"][prop_name] = stats_pool[key]
        else:
            info["Inputs"][prop_name] = {
                "max": 1.0, "mean": None, "min": 0.0, "norm": None, "std": None
            }
        info["Inputs"][prop_name]["index"] = idx

    save_yaml(info, dest_dir / "info.yaml")

    # Apply transforms (e.g., SDF) like in prepare_dataset_for_sequence
    transforms = get_transforms(reduce_to_2D=False, inputs="".join(step3_inputs))
    tensor_transform = ToTensorTransform()
    print(f"Step3 inputs: {step3_inputs}, step3 outputs: {step3_outputs}")
    print(f"Transforms for Step 3: {[t.__class__.__name__ for t in transforms.transforms]}")

    step2_expanded_inputs = [name for k in step2_inputs for name in expand_property_names(k)]
    step3_expanded_inputs = [name for k in step3_inputs for name in expand_property_names(k)]
    step2_index = {name: i for i, name in enumerate(step2_expanded_inputs)}

    inputs_dir = dest_dir / "Inputs"
    for input_file in inputs_dir.iterdir():
        x_tensor = torch.load(input_file)
        if x_tensor.dim() not in (3, 4):
            raise ValueError(f"Unexpected input tensor shape in {input_file}: {x_tensor.shape}")

        # Preserve time dimension if present: (C, T, H, W)
        if x_tensor.dim() == 4:
            time_slices = []
            for t in range(x_tensor.shape[1]):
                x_slice = x_tensor[:, t, :, :]
                base_dict = {}
                for name in step2_expanded_inputs:
                    if name in step2_index:
                        base_dict[name] = x_slice[step2_index[name]]
                    else:
                        base_dict[name] = torch.zeros_like(x_slice[0])
                if "SDF" in step3_expanded_inputs and "SDF" not in base_dict:
                    if "Material ID" in base_dict:
                        base_dict["SDF"] = base_dict["Material ID"].clone()
                    else:
                        base_dict["SDF"] = torch.zeros_like(x_slice[0])
                loc_hp = load.get_hp_location(base_dict) if "Material ID" in base_dict else None
                base_dict = transforms(base_dict, loc_hp=loc_hp)
                out_dict = {}
                for name in step3_expanded_inputs:
                    if name in base_dict:
                        out_dict[name] = base_dict[name]
                    else:
                        out_dict[name] = torch.zeros_like(x_slice[0])
                time_slices.append(tensor_transform(out_dict))
            x_tensor = torch.stack(time_slices, dim=1)
        else:
            base_dict = {}
            for name in step2_expanded_inputs:
                if name in step2_index:
                    base_dict[name] = x_tensor[step2_index[name]]
                else:
                    base_dict[name] = torch.zeros_like(x_tensor[0])
            if "SDF" in step3_expanded_inputs and "SDF" not in base_dict:
                if "Material ID" in base_dict:
                    base_dict["SDF"] = base_dict["Material ID"].clone()
                else:
                    base_dict["SDF"] = torch.zeros_like(x_tensor[0])
            loc_hp = load.get_hp_location(base_dict) if "Material ID" in base_dict else None
            base_dict = transforms(base_dict, loc_hp=loc_hp)
            out_dict = {}
            for name in step3_expanded_inputs:
                if name in base_dict:
                    out_dict[name] = base_dict[name]
                else:
                    out_dict[name] = torch.zeros_like(x_tensor[0])
            x_tensor = tensor_transform(out_dict)

        torch.save(x_tensor, input_file)

    # Generate args.yaml
    args = {
        "dataset": dataset_name,
        "inputs": expanded_step3_inputs,
        "outputs": [n for k in step3_outputs for n in expand_property_names(k)]
    }

    if use_vel_model:
        # Tag inputs predicted by velocity model
        for key in step1_outputs:
            if key in step3_map:
                args["inputs"][step3_map[key]] += f" - predicted by '{vel_model_dir.name}'"

    save_yaml(args, dest_dir / "args.yaml")


def crop_and_merge_tensors(
    inputs: torch.Tensor,
    velocity: torch.Tensor,
    vel_out_map: Dict[str, int],
    step3_in_map: Dict[str, int]
) -> torch.Tensor:
    """Center-crop inputs to match velocity dimensions and inject velocity channels."""
    h_req, w_req = velocity.shape[-2:]
    h_curr, w_curr = inputs.shape[-2:]

    # Calculate center crop offsets
    dy, dx = (h_curr - h_req) // 2, (w_curr - w_req) // 2

    # Clone to decouple memory and perform crop
    merged = inputs[:, dy : dy + h_req, dx : dx + w_req].clone()

    for name, src_idx in vel_out_map.items():
        if name in step3_in_map:
            merged[step3_in_map[name]] = velocity[src_idx]

    return merged
