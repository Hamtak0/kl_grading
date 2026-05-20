import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Locks down all sources of randomness for absolute reproducibility.
    """
    # 1. Python base library
    random.seed(seed)
    
    # 2. NumPy operations
    np.random.seed(seed)
    
    # 3. PyTorch CPU and GPU tensors
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # 4. CuDNN determinism flags (forces GPU to use deterministic algorithms)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False