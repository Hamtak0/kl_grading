# Knee Osteoarthritis with KL-Grading

## About

  Two-stage deep learning pipeline for the autonomous detection and severity grading of knee osteoarthritis from bilateral standing anterior to posterior view.

## Overview Project Tree

```md
.
├── configs
│   ├── config.toml
│   └── inference.toml
├── core
│   ├── classification
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── optuna_search.py
│   │   ├── test.py
│   │   └── train.py
│   ├── data
│   │   ├── __init__.py
│   │   └── dataset_handler.py
│   ├── detection
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── test.py
│   │   └── train.py
│   └── __init__.py
├── dataset
│   ├── bilateral_standing_AP
│   ├── cached_crops
│   ├── KLGrade_label_with_5fold.xlsx
│   ├── landmarks
│   └── SO_landmarks
├── devenv.lock
├── devenv.nix
├── devenv.yaml
├── main.py
├── misc
│   ├── __init__.py
│   ├── cache_dataset.py
│   ├── drag_data.py
│   └── param_model.py
├── pyproject.toml
├── README.md
├── results
│   ├── classification
│   ├── inference
│   └── object_detection
├── scripts
│   ├── pipeline_classification.py
│   └── pipeline_object_detection.py
├── utils
│   ├── __init__.py
│   ├── config.py
│   ├── cropping.py
│   ├── dicom_cut.py
│   ├── logger.py
│   ├── metrics.py
│   ├── resize.py
│   ├── seed_setup.py
│   ├── transform.py
│   └── visualize_rater.py
└── uv.lock
```

## Usage Guide

### 0. Prerequisites & preparation

  Before running any scripts, ensure the configuration file is properly set up.

  ```bash
  uv sync
  ``` 

### 1. Object Detection

  **Goal**: Train the Faster R-CNN model to locate left and right knee joints from full bilateral standing anterior to posterior x-rays.

  ```bash
  uv run python -m scripts.pipeline_object_detection
  ```

  **Expected output**: A folder `./results/object_detection/{timestamp}_{strategy}` containing the `.pth` weights, TensorBoard runs, and visual testing images.

### 2. Classification

  **Goal**: Run an optuna hyperparameter search. Dynamically load the resulting JSON, and train the DenseNet model to classify knee joint conditions.

  ```bash
  uv run python -m scripts.pipeline_classification
  ```

  **Expected output**: A folder `./results/classification/{timestamp}_{strategy}_{target}_{mode}_{loss}` containing `best_trial_config.json`, the trained model in folds, TensorBoard runs, visual validating and testing images, and `metrics/` which contains confusion matrics and csv.

### 3. Inference

  **Goal**: Run the inference pipeline to classify knee joint conditions from the both object detection and classification trained models.

  ```bash
  uv run main.py
  ```

  **Expected output**: A folder `./results/inference/{rcnn_ts}_{cls_ts}_{strategy}_{target}_{mode}_{loss}` containing labelled images, confusion matrix, and csv file.
