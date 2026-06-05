import os
import torch
from pathlib import Path
from tqdm import tqdm
from core.data.dataset_handler import KLGradingDataset, ClassifierDataset
from utils.seed_setup import set_seed

def build_tensor_cache():
    print("Initializing base dataset for caching...")
    root_dir = "./dataset"
    
    # Load raw pixels (transforms=None ensures correct black padding)
    full_dataset = KLGradingDataset(
        root=root_dir,
        transforms=None,
        include_grades=True,
        grade_path="./dataset/KLGrade_label_with_5fold.xlsx"
    )
    
    wrapper = ClassifierDataset(full_dataset)
    
    # Create the cache directory
    cache_dir = Path("./dataset/cached_crops")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting and caching {len(wrapper)} knees...")
    
    for i in tqdm(range(len(wrapper)), desc="Caching Tensors"):
        crop, label, oa, pid = wrapper[i]
        
        # Save as a dictionary containing the ready-made tensor and label
        cache_data = {
            "crop": crop.clone().detach(),  # raw padded tensor
            "label": label.clone().detach(),
            "oa": oa.clone().detach(),
            "patient_id": pid
        }
        
        save_path = cache_dir / f"{pid}.pt"
        torch.save(cache_data, save_path)
        
    print(f"\nSuccess! All knees cached directly to {cache_dir}")

if __name__ == "__main__":
    set_seed(42)  # Ensure reproducibility for any random operations during caching
    build_tensor_cache()