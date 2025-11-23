"""
Unified interface for loading alignment training dataloaders.
Supports both VOC and robot datasets.
"""

from typing import Tuple
from torch.utils.data import DataLoader

from .dataset_voc import get_alignment_dataloader as get_voc_dataloader
from .dataset_robot import get_alignment_dataloader as get_robot_dataloader


def get_alignment_dataloader(
    dataset: str, 
    batch_size: int = 12, 
    apply_mask: bool = True, 
    corruption_severity: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Unified interface for loading alignment training dataloaders.
    
    Args:
        dataset (str): Dataset type, either "voc" or "robot"
        batch_size (int, optional): The batch size of the dataloader. Defaults to 12.
        apply_mask (bool, optional): Whether to apply mask overlay. Defaults to True. (Note: this is ignored for robot data)
        corruption_severity (int, optional): Severity of corruption transforms (0-5). Defaults to 0. (Note: this is ignored for robot data)
    
    Returns:
        Tuple[DataLoader, DataLoader]: The train and validation loader respectively.
        
    Raises:
        ValueError: If dataset is not "voc" or "robot"
    
    Note:
        Images are returned at their original resolution. The RAE model handles resizing internally.
    """
    if dataset == "voc":
        return get_voc_dataloader(
            batch_size=batch_size,
            apply_mask=apply_mask,
            corruption_severity=corruption_severity,
        )
    elif dataset == "robot":
        return get_robot_dataloader(
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unsupported dataset: '{dataset}'. Must be 'voc' or 'robot'.")
