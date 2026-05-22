"""
Dataset class for the KL grading dataset.
Special thanks to https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""

import os
import numpy as np
import json
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from utils.dicom_cut import read_dicom_image, mm_to_pixel
from utils.cropping import cropping_points, center_to_corners
from utils.resize import extract_uniform_crop

class KLGradingDataset(Dataset):
    def __init__(self, root, transforms=None, include_grades=True, grade_path="./dataset/grades.json"):
        self.root = root
        self.transforms = transforms
        self.include_grades = include_grades

        dicom_dir = Path(root) / "bilateral_standing_AP"
        bounding_dir = Path(root) / "landmarks"

        all_dicom_files = os.listdir(dicom_dir) if os.path.exists(dicom_dir) else []
        all_bounding_files = os.listdir(bounding_dir) if os.path.exists(bounding_dir) else []

        dicom_map = {f.split(".")[0]: f for f in all_dicom_files}
        bounding_map = {f.split(".")[0]: f for f in all_bounding_files}

        paired_ids = set(dicom_map).intersection(set(bounding_map))

        if self.include_grades:
            try:
                df = pd.read_excel(grade_path, dtype={"ID": str})
                excel_map = set(df['ID'].astype(str).tolist())

                valid_ids = paired_ids.intersection(excel_map)

                self.labels_dict = {}
                self.patient_to_fold = {}

                # print(df.head(20))  # Debug: Check the contents of the DataFrame
                for _, row in df.iterrows():
                    pid = str(row['ID'])
                    if pid not in valid_ids: 
                        continue
                    side = row['Side']
                    kl = row['KL']
                    oa = row['OA']
                    fold = row['fold']

                    if pid not in self.labels_dict:
                        self.labels_dict[pid] = {"L": [-1, -1], "R": [-1, -1]} 
                    self.labels_dict[pid][side] = [kl, oa]
                    self.patient_to_fold[pid] = fold

            except Exception as e:
                raise Exception(f"Failed to load grades from {grade_path}: {e}")
        else:
            valid_ids = paired_ids

        self.patient_ids = sorted(list(valid_ids))
        self.dicom = [dicom_map[pid] for pid in self.patient_ids]
        self.bounding = [bounding_map[pid] for pid in self.patient_ids]

    def _get_pixel_box(self, ds, crop_data):
        corners_mm = center_to_corners(*crop_data)
        corners_px = np.array([mm_to_pixel(ds, *point) for point in corners_mm])

        xmin = np.min(corners_px[:, 0])
        xmax = np.max(corners_px[:, 0])
        ymin = np.min(corners_px[:, 1])
        ymax = np.max(corners_px[:, 1])

        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        dicom_path = os.path.join(self.root, "bilateral_standing_AP", self.dicom[idx])
        bounding_path = os.path.join(self.root, "landmarks", self.bounding[idx])

        patient_id = self.bounding[idx].split(".")[0]

        ds, image_array = read_dicom_image(dicom_path)
        # Convert grayscale array to a 3-channel PyTorch tensor (C, H, W)
        # Faster R-CNN natively expects 3 channels
        img_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

        with open(bounding_path, 'r') as f:
            data = json.load(f)

        left_crop_mm, right_crop_mm = cropping_points(data)

        left_box = self._get_pixel_box(ds, left_crop_mm)
        right_box = self._get_pixel_box(ds, right_crop_mm)

        boxes_array = [left_box, right_box]
        boxes = torch.tensor(boxes_array, dtype=torch.float32)

        # create labels (1 for left knee and 2 for right knee)
        labels = torch.tensor([1, 2], dtype=torch.int64)

        kl_list, oa_list = [], []
        if self.include_grades and hasattr(self, 'labels_dict'):
            # syntax: df.get(key, default=None) 
            patient_grades = self.labels_dict.get(patient_id, {"L": [-1, -1], "R": [-1, -1]})
            try:
                for label in labels:
                    if label.item() == 1:  # Left knee
                        grades = patient_grades.get("L", [-1, -1])
                    elif label.item() == 2:  # Right knee
                        grades = patient_grades.get("R", [-1, -1])
                    # grades = [kl, oa]
                    kl_list.append(grades[0])
                    oa_list.append(grades[1])
                    
            except Exception as e:
                raise Exception(f"Error while extracting KL grades for patient {patient_id}: {e}")
        else:
            # none both left and right side 
            kl_list = [-1, -1]
            oa_list = [-1, -1]

        # metadata for torchvision see https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset
        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((2, ), dtype=torch.int64)

        img = tv_tensors.Image(img_tensor)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = labels
        target["image_id"] = torch.tensor([image_id])
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["kl_grades"] = torch.tensor(kl_list, dtype=torch.long)
        target["osteoarthritis"] = torch.tensor(oa_list, dtype=torch.long)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.dicom)

class ClassifierDataset(Dataset):
    """
    Wraps your existing KLGradingDataset. 
    It intercepts the full X-ray, crops out the knees using the ground truth boxes, 
    and yields individual 224x224 images paired with their KL grade.
    """
    def __init__(self, base_subset):
        self.base_subset = base_subset
        self.samples = []
        
        # Unpack the dataset to create a flat list of individual knees
        for i in range(len(base_subset)):
            _, target = base_subset[i]
            boxes = target['boxes'].numpy()
            kl_grades = target['kl_grades'].numpy() 
            oa_grades = target['osteoarthritis'].numpy()
            
            original_idx = target['image_id'].item()  # Get the original index from the target
            if isinstance(self.base_subset, Subset):
                dicom_filename = self.base_subset.dataset.dicom[original_idx] 
            else:
                dicom_filename = self.base_subset.dicom[original_idx]

            patient_id = dicom_filename.split(".")[0]

            for side_idx, (box, kl_grade, oa_grades) in enumerate(zip(boxes, kl_grades, oa_grades)):
                if kl_grade == -1 or oa_grades == -1:
                    continue  # Skip if KL or OA grade is missing for this knee

                # This block creates each samples with a unique patient ID that includes the side (L/R) for better tracking
                side = "_L" if target['labels'][side_idx].item() == 1 else "_R"

                # We store the index and the specific box/grade so __getitem__ can crop it later
                self.samples.append({
                    "base_index": i,
                    "box": box,
                    "kl_grade": kl_grade,
                    "oa_grade": oa_grades,
                    "patient_id": f"{patient_id}{side}",
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, only_gray=False):
        sample_info = self.samples[idx]
        img_tensor, _ = self.base_subset[sample_info["base_index"]]
        
        # Crop exactly to the ground truth box
        crop_tensor = extract_uniform_crop(img_tensor, sample_info["box"])

        # Because right now using ResNet which requires [3, 224, 224] therefore
        # the only_gray option doesn't need to be used
        if only_gray:
            crop_tensor = crop_tensor[0:1, :, :] # keep only grayscale
        
        # Classifiers expect long (integer) tensors for the target labels
        kl_grade = torch.tensor(sample_info["kl_grade"], dtype=torch.long)
        oa_grade = torch.tensor(sample_info["oa_grade"], dtype=torch.long)
        
        return crop_tensor, kl_grade, oa_grade, sample_info["patient_id"]  # Return patient ID for potential debugging or tracking

class CachedKneeDataset(Dataset):
    """
    Reads pre-processed .pt crops instantly from disk. 
    Completely bypasses DICOM decompression and JSON parsing overhead.
    """
    def __init__(self, cache_dir="./dataset/cached_crops", root="./dataset", grade_path="./dataset/KLGrade_label_with_5fold.xlsx"):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(list(self.cache_dir.glob("*.pt")))

        if self.files:
            # print(f"Found {len(self.files)} cached crop files in {self.cache_dir}.")
            self.use_cache = True

            df = pd.read_excel(grade_path, dtype={"ID": str})
            self.patient_to_fold = dict(zip(df['ID'], df['fold']))
        else:
            # print(f"Cache not found in {self.cache_dir}. falling back to original dataset.")
            self.use_cache = False

            full_dataset = KLGradingDataset(root=root, transforms=None, include_grades=True, grade_path=grade_path)
            self.fallback_dataset = ClassifierDataset(full_dataset)
            self.patient_to_fold = self.fallback_dataset.patient_to_fold
        
    def __len__(self):
        if self.use_cache:
            return len(self.files)
        else:
            return len(self.fallback_dataset)
        
    def __getitem__(self, idx):
        if self.use_cache:
            # binary load is near-instantaneous and bypasses CPU bottlenecks
            data = torch.load(self.files[idx], weights_only=True)
            return data["crop"], data["label"], data["oa"], data["patient_id"]
        else:
            return self.fallback_dataset[idx]

    @property
    def samples(self):
        if self.use_cache:
            # Build the mock list exactly once and store it in memory to save CPU time
            if not hasattr(self, '_mock_samples'):
                self._mock_samples = []
                for f in self.files:
                    # Load the binary tensor dictionary to extract the exact label
                    data = torch.load(f, weights_only=True)
                    self._mock_samples.append({
                        "patient_id": f.stem,
                        "kl_grade": data["label"].item(),
                        "oa_grade": data["oa"].item()
                    })
            return self._mock_samples
        else:
            return self.fallback_dataset.samples

class TransformWrapper(Dataset):
    """
    Wraps a dataset subset to apply PyTorch transforms dynamically.
    This guarantees we crop and pad the image first and normalize second.
    """
    def __init__(
            self,
            subset,
            # transform=None,
            mode="BOTH"
        ):
        self.subset = subset
        # self.transform = transform
        self.mode = mode

    def __getitem__(self, idx):
        crop, label, oa, pid = self.subset[idx]
        """* Note: handle the tranform inside the training loop instead (cpu bottleneck)
        if self.transform is not None:
            # Ensure it is a TV_Tensor so v2 transforms work perfectly
            crop = tv_tensors.Image(crop)
            crop = self.transform(crop)
        """
        is_left = pid.endswith("_L")
        is_right = pid.endswith("_R")

        if self.mode == "LEFT" and is_right:
            crop = torch.flip(crop, dims=[2]) 
        elif self.mode == "RIGHT" and is_left:
            crop = torch.flip(crop, dims=[2])

        return crop, label, oa, pid

    def __len__(self):
        return len(self.subset)

class Fold_Handler:
    """
    A utility class to manage dataset splits based on the predefined Excel file.
    """
    def __init__(self, base_dataset):
        try:
            if not hasattr(base_dataset, 'patient_to_fold'):
                raise ValueError("The provided dataset does not contain fold information.")
            self.patient_to_fold = base_dataset.patient_to_fold
        except Exception as e:
            raise Exception(f"Failed to initialize Fold_Handler: {e}")

    def get_fold(self, patient_id):
        if patient_id not in self.patient_to_fold:
            raise ValueError(f"Patient {patient_id} not found in the fold mapping.")
        return self.patient_to_fold[patient_id]

    def get_test_fold(self):
        return 0

    def get_cv_folds(self):
        all_unique_folds = set(self.patient_to_fold.values())
        cv_folds = all_unique_folds - {self.get_test_fold()}
        return sorted(list(cv_folds))