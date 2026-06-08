import numpy as np
import torch
import yaml
from pathlib import Path
from typing import List, Dict, Annotated, Any, Optional, Union, Literal, Type, TypeVar, Tuple
from pydantic import BaseModel, BeforeValidator, Field, ConfigDict, ValidationError

# --- Helper Types & Validators ---

def force_string(v: Any) -> str:
    """Coerces any input to string."""
    return str(v)

CoercedString = Annotated[str, BeforeValidator(force_string)]
T = TypeVar("T", bound=BaseModel)

# --- Configuration Models ---

class Datapoints(BaseModel):
    """Defines split sizes for datasets."""
    validation: List[int]
    test: List[int]
    train: List[int]

class GeneralSettings(BaseModel):
    """General training settings."""
    epochs: int
    visualize: bool
    visualize_interval: Optional[int]
    model_path: Path

class ReduceLROnPlateauConfig(BaseModel):
    """Configuration for ReduceLROnPlateau scheduler."""
    type: Literal["ReduceLROnPlateau"]
    init_lr: float
    mode: str = "min"
    factor: float
    patience: int
    threshold: float
    min_lr: float
    early_stop_patience: Optional[int] = None

class StepLRConfig(BaseModel):
    """Configuration for StepLR scheduler."""
    type: Literal["StepLR"]
    init_lr: float
    step_size: int
    gamma: float
    early_stop_patience: Optional[int] = None
    

class ConstantLRConfig(BaseModel):
    """Configuration for constant learning rate (no scheduler)."""
    type: Literal["constant"]
    init_lr: float
    early_stop_patience: int

SchedulerConfig = Annotated[Union[ReduceLROnPlateauConfig, StepLRConfig, ConstantLRConfig], Field(discriminator='type')]

class NetworkParameters(BaseModel):
    inputs: List[CoercedString]
    outputs: List[CoercedString]
    batchsize: int
    stride: int
    skip_per_dir: int
    len_box: int
    train_loss: str
    bool_cutouts: bool
    optimizer_switch: bool
    optimizer: str
    lr: float
    activation: str

class UNetParameters(NetworkParameters):
    """Hyperparameters specific to UNet architecture."""
    network: Literal["unet"]
    kernel_size: int
    depth: int
    dilation: int
    norm: Optional[str]
    repeat_inner: bool
    init_features: int
    

class RNNParameters(NetworkParameters):
    """Hyperparameters specific to RNN architecture."""
    network: Literal["rnn"]
    enc_conv_features: List[int] = [32, 64, 128, 256, 512]
    dec_conv_features: List[int] = [512, 256, 128, 64, 32]
    enc_kernel_sizes: List[int] = [5, 5, 5, 5, 5]
    dec_kernel_sizes: List[int] = [5, 5, 5, 5, 5]
    time_steps_to_predict: List[List[int]]
    rnn_num_layers: int
    max_simulation_timestep: int = 55

ModelConfig = Annotated[Union[UNetParameters, RNNParameters], Field(discriminator='network')]

class HoptParameters(BaseModel):
    """Search space for hyperparameter optimization."""
    network: List[str]
    inputs: List[List[CoercedString]]
    outputs: List[List[CoercedString]]
    batchsize: List[int]
    kernel_size: List[int]
    depth: List[int]
    init_features: List[int]
    stride: List[int]
    dilation: List[int]
    norm: List[Optional[str]]
    repeat_inner: List[bool]
    skip_per_dir: List[int]
    len_box: List[int]
    train_loss: List[str]
    bool_cutouts: List[bool]
    optimizer_switch: List[bool]
    optimizer: List[str]
    lr: List[float]
    activation: List[str]

class MLStepConfig(BaseModel):
    """Configuration for ML pipeline steps."""
    previous_results: Optional[Path] = None
    datapoints: Datapoints
    general: GeneralSettings
    scheduler: SchedulerConfig
    model_parameters: ModelConfig
    hopt_parameters: Optional[HoptParameters] = None
    
class Streamlines(BaseModel):
    """Streamline simulation parameters."""
    samples: int
    steps: int
    diffusion_scale: float
    diffusion_base: float

class DirectSolver(BaseModel):
    """Direct solver parameters."""
    samples: int
    steps: int

class TimeSeriesParam(BaseModel):
    """Time-dependent physical parameters."""
    time_unit: str
    values: Dict[float, float]

class PhysicalParameters(BaseModel):
    """Physical constants and field parameters."""
    resolution_m: float
    duration_years: float
    porosity_frac: float
    rock_density_kg_per_m3: float
    rock_specific_heat_J_per_kgK: float
    water_density_kg_per_m3: float
    water_specific_heat_J_per_kgK: float
    thermal_conductivity_dry_W_per_mK: float
    thermal_conductivity_wet_W_per_mK: float
    thickness_aquifer_m: float
    longitudinal_dispersivity_m: float
    transverse_dispersivity_h_m: float
    ambient_temperature_C: float
    injection_temperature_C: TimeSeriesParam
    injection_rate_m3_per_s: TimeSeriesParam

class SimulationStepConfig(BaseModel):
    """Configuration for physics simulation step."""
    streamlines: Streamlines
    directsolver: DirectSolver
    physical_parameters: PhysicalParameters

class GeneralConfiguration(BaseModel):
    """Aggregated configuration for all pipeline steps."""
    step1: MLStepConfig
    step2: SimulationStepConfig
    step3: MLStepConfig

class RunConfiguration(BaseModel):
    """Meta-configuration for the execution environment."""
    run_name: str
    dataset: str
    seed: int
    device: str
    use_velocity_model: bool
    overfit: bool = False
    overfit_on: Optional[int]
    pipeline: List[Dict[str, str]]

class Paths(BaseModel):
    """Directory paths for data and results."""
    datasets_raw: Path
    datasets_prep: Path
    results: Path

class AppConfig(BaseModel):
    """Root configuration object."""
    run_configuration: RunConfiguration
    general_configuration: GeneralConfiguration
    paths: Paths

# --- Logic & Parsers ---

def load_config(yaml_path: Union[str, Path], model_cls: Type[T]) -> T:
    """Generic loader for YAML to Pydantic models."""
    try:
        with open(yaml_path, 'r') as f:
            return model_cls(**yaml.safe_load(f))
    except (ValidationError, FileNotFoundError, yaml.YAMLError) as e:
        raise ValueError(f"Failed to load {model_cls.__name__} from {yaml_path}: {e}") from e

def parse_config(yaml_file_path: str) -> AppConfig:
    """Parses main application configuration."""
    return load_config(yaml_file_path, AppConfig)

def parse_model_config(yaml_file_path: str) -> ModelConfig:
    """Parses specific model configuration."""
    return load_config(yaml_file_path, ModelConfig)

def convert_injection_config(
    data: TimeSeriesParam, 
    steps_count: int, 
    time_end_years: float, 
    device: Union[str, torch.device]
) -> Tuple[torch.Tensor, int]:
    """Interpolates time-series injection data into a tensor for simulation."""
    if data.time_unit != "year":
        raise ValueError(f"Unsupported time unit: {data.time_unit}. Only 'year' is supported.")
    if steps_count <= 0:
        raise ValueError("Step count must be positive.")
    if time_end_years <= 0:
        raise ValueError("Time end must be positive.")

    # Sort control points by time
    xp = sorted(data.values.keys())
    fp = [data.values[k] for k in xp]
    period = xp[-1]

    # Vectorized interpolation
    t_eval = np.linspace(0, time_end_years, steps_count)
    t_cycle = t_eval % period
    cycle_values = np.interp(t_cycle, xp, fp)

    # Calculate seasonal cycles
    seasonal_cycles = int(period / (time_end_years / steps_count)) if steps_count > 0 else 0
    
    return torch.from_numpy(cycle_values).float().to(device), seasonal_cycles
