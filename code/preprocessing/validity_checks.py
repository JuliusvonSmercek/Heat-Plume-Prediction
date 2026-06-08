from torch.utils.data import DataLoader
from typing import Dict

def receptive_field_is_sufficient(receptive_field: int, dataloaders: Dict[str, DataLoader],  time_step_interval: float=5.0, resolution: float=1.0) -> bool:
    """
    Check if the receptive field is sufficient for the input size of the dataloaders.

    Args:
        max_rf (int): The maximum receptive field of the model.
        dataloaders (Dict[str, DataLoader]): A dictionary containing the dataloaders for 'train', 'val', and 'test'.

    Returns:
        bool: True if the receptive field is sufficient, False otherwise.
    """
    max_velocity = get_max_velocity(dataloaders) # in x-richtung oder integral über breite receptivefield
    
    required_receptive_field = int((max_velocity * time_step_interval) / resolution)
    
    return receptive_field >= required_receptive_field       

def get_max_velocity(dataloaders: Dict[str, DataLoader]) -> float:
    
    # TODO: Iterate over all dataloader to find the maximum velocity in the input
    
    return 1.0