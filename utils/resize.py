import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T
import numpy as np

def extract_uniform_crop(image_tensor: torch.Tensor, box: tuple[float, float, float, float] | list[float] | np.ndarray , target_size: int = 512) -> torch.Tensor:
    """
    Takes a full image tensor and a bounding box. 
    Returns a perfectly undistorted target_size x target_size tensor.
    """
    xmin, ymin, xmax, ymax = map(int, box)
    crop = image_tensor[:, ymin:ymax, xmin:xmax]
    
    _, h, w = crop.shape
    max_dim = max(h, w)
    
    gap_height = max_dim - h
    pad_top = gap_height // 2
    pad_bottom = gap_height - pad_top
    
    gap_width = max_dim - w
    pad_left = gap_width // 2
    pad_right = gap_width - pad_left
    
    squared_crop = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    resize_transform = T.Resize((target_size, target_size), antialias=True)
    final_crop = resize_transform(squared_crop)

    return final_crop 