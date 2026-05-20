from torchvision.transforms import v2 as T
import torch
import numpy as np
from typing import Any

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transform(rotation: float = 10, jitter: float = 0.3, sharpness: float = 3.0, is_train: bool = False) -> T.Compose:
    transforms = [
        # https://pytorch.org/hub/pytorch_vision_resnet/
        T.ToImage(),
        T.ToDtype(torch.float, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.ToPureTensor()
    ]
    if is_train:
        transforms.extend([
            T.RandomAffine(degrees=rotation, translate=(0.05, 0.05), scale=(0.8, 1.2)),
            T.RandomAdjustSharpness(sharpness_factor=sharpness),
            T.ColorJitter(brightness=jitter, contrast=jitter)
        ])
    return T.Compose(transforms)

def unnormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Takes a normalized PyTorch batch tensor (B, C, H, W)
    and un-normalizes it back to a viewable numpy array [0, 1] formatted as (B, H, W, C).
    """
    images_np = tensor.cpu().numpy().transpose(0, 2, 3, 1)
    
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    
    images_np = std * images_np + mean
    return np.clip(images_np, 0, 1)

def collate_fn(batch: list[tuple[Any, ...]]) -> tuple[Any, ...]:
    return tuple(zip(*batch))
