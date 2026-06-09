from pathlib import Path
import yaml
from typing import List, Union
from torch import Tensor
import numpy as np
import h5py
import re
import utils.logging as log
import os

def read_cla(path:str):
    clas = load_yaml(path / "command_line_arguments.yaml")
    for path_typed_cla in ["data_prep", "data_raw", "model", "destination"]:
        try:
            if clas[path_typed_cla] is not None:
                clas[path_typed_cla] = Path(clas[path_typed_cla])
        except KeyError:
            continue
    clas["destination"] = path

    return clas

def get_data_prep_path(prep_dir: Path, inputs: str, outputs: str, data_raw: Path) -> Path:
    return prep_dir / data_raw.name / f"inputs_{inputs}_outputs_{outputs}"

def make_data_prep_dir(data_prep: Path):
    log.info(f"Dataset_pre path: {data_prep}")
    data_prep.mkdir(parents=True, exist_ok=True)
    (data_prep / "Inputs").mkdir(parents=True, exist_ok=True)
    (data_prep / "Labels").mkdir(parents=True, exist_ok=True)

def check_model_avail(args: dict):
    model_dir = Path(args["model"])
    if not (model_dir / "model.pt").exists():
        raise FileNotFoundError(f"model.pt not found in {model_dir}")
    print(list(model_dir.iterdir()))
    if not (model_dir / "config.yaml").exists():
        raise FileNotFoundError(f"config.yaml not found in {model_dir}")

def load_yaml(path: Path, **kwargs) -> dict:
    with open(path, "r") as file:
        args = yaml.safe_load(file, **kwargs)
    return args

def load_time_steps(path: Path) -> List[float]:
    with h5py.File(path, 'r') as file:
        times = [
            int(re.match(r"\s*(\d+)", key).group(1))
            for key in file.keys()
        ]
    
    return times

def load_time_steps_full_str(path: Path) -> List[str]:
    with h5py.File(path, 'r') as file:
        times = [ str(key) for key in file.keys() ]
    
    return times

# Convert tensors to Python-native types
def convert_to_python_datatypes(data):
    if isinstance(data, Tensor):
        return data.item() if data.numel() == 1 else data.tolist()
    elif isinstance(data, (np.ndarray, np.generic)):
        return data.item() if np.isscalar(data) else data.tolist()
    elif isinstance(data, dict):
        return {k: convert_to_python_datatypes(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_python_datatypes(v) for v in data]
    else:
        return data


def save_yaml(args:dict, destination_file):
    with open(destination_file, "w") as file:
        tmp = args.copy()
        for arg in args.keys():
            try:
                for info in arg.keys():
                    tmp[info] = path_to_str(arg[info])
            except:
                tmp[arg] = path_to_str(args[arg])
        # Convert tensors to Python-native types
        tmp = convert_to_python_datatypes(tmp)
        # Save to YAML file
        yaml.dump(tmp, file, default_flow_style=False)

def path_to_str(arg: Union[Path, str]) -> str:
    '''if arg a Path object, convert to string'''
    if isinstance(arg, Path):
        return str(arg)
    return arg

def get_run_ids_from_prep(dir: Path) -> List[int]:
    run_ids = []
    for file in dir.iterdir():
        if file.suffix == ".pt":
            run_ids.append(int(file.stem.split("_")[-1]))
            # log.info(f"Found run_id {run_ids[-1]}")
    run_ids.sort()
    return run_ids

# OTHER UTILS
def is_empty(path:Path):
    return len(list(path.iterdir())) == 0