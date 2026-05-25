import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score

from dataset_handler import CachedKneeDataset, Fold_Handler, TransformWrapper
from models import Classification_ResNet, Classification_DenseNet
from utils.logger import setup_logger
from utils.seed_setup import set_seed
from utils.transform import get_transform
from utils.config import load_toml_config

def objective(trial: int = 15, mode: str = "BOTH", base_epochs: int = 50, target: str = "OA"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameter Search Space
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True)
    eta_min = trial.suggest_float("eta_min", 1e-7, 1e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    # To handle the dedicated GPU memory
    physical_batch_size = 2
    accumulation_steps = max(1, batch_size // physical_batch_size)
    
    env_cfg = load_toml_config("configs/config.toml")
    cv_classifier_dataset = CachedKneeDataset(
        cache_dir=env_cfg["dataset"]["cache_dir"],
        root=env_cfg["dataset"]["root_dir"],
        grade_path=env_cfg["dataset"]["excel_file"]
    )
    fold_handler = Fold_Handler(cv_classifier_dataset)

    cv_folds = fold_handler.get_cv_folds()
    val_fold = cv_folds[0]
    train_folds = cv_folds[1:]

    gpu_train_transform = get_transform(rotation=10, jitter=0.3, sharpness=2, is_train=True).to(device)
    gpu_val_transform = get_transform(rotation=0, jitter=0, sharpness=0, is_train=False).to(device)

    train_idx, val_idx = [], []
    for i, sample in enumerate(cv_classifier_dataset.samples):
        base_id = sample["patient_id"].split("_")[0]
        assigned_fold = fold_handler.get_fold(base_id)
        if assigned_fold == val_fold:
            val_idx.append(i)
        elif assigned_fold in train_folds:
            train_idx.append(i)

    train_subset = Subset(cv_classifier_dataset, train_idx)
    val_subset = Subset(cv_classifier_dataset, val_idx)

    train_dataset = TransformWrapper(train_subset, mode=mode)
    val_dataset = TransformWrapper(val_subset, mode=mode)
    
    data_loader_train = DataLoader(
        train_dataset,
        batch_size=physical_batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True,
    )
    data_loader_val = DataLoader(
        val_dataset,
        batch_size=physical_batch_size,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
    )
    
    if target == "KL":
        # model = Classification_ResNet(num_classes=5).to(device)
        model = Classification_DenseNet(num_classes=1).to(device)
        train_labels = [cv_classifier_dataset.samples[i]["kl_grade"] for i in train_idx]
        class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(5), y=train_labels)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        # criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)
        criterion = nn.MSELoss(reduction='none')
    elif target == "OA":
        model = Classification_DenseNet(num_classes=2).to(device)
        train_labels = [cv_classifier_dataset.samples[i]["oa_grade"] for i in train_idx]
        class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=train_labels)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=base_epochs, eta_min=eta_min)
    
    scaler = torch.amp.GradScaler('cuda')
    best_val_score = 0.0
    
    for epoch in range(base_epochs):
        model.train()

        optimizer.zero_grad()
        for step, (images, kl, oa, _) in enumerate(data_loader_train): # images, kl, oa, pid
            if target == "KL":
                labels = kl.to(device)
            elif target == "OA":
                labels = oa.to(device)
            images = gpu_train_transform(images.to(device))

            # optimizer.zero_grad()

            outputs = model(images)
            with torch.amp.autocast('cuda'):
                if target == "KL":
                    outputs = outputs.squeeze(1)
                    unweighted_loss = criterion(outputs, labels.float())
                    weighted_sum = (unweighted_loss * weights_tensor[labels.long()]).sum()
                    raw_loss = weighted_sum / weights_tensor[labels.long()].sum()
                elif target == "OA":
                    raw_loss = criterion(outputs, labels)
                loss = raw_loss / accumulation_steps
            
            scaler.scale(loss).backward()
            # loss.backward()
    
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(data_loader_train):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # optimizer.step()
    
        model.eval()
        all_val_labels, all_val_preds = [], []
        with torch.no_grad():
            for images, kl, oa, _ in data_loader_val:
                if target == "KL":
                    labels = kl.to(device)
                elif target == "OA":
                    labels = oa.to(device)
                images = gpu_val_transform(images.to(device))
    
                outputs = model(images)
                if target == "KL":
                    outputs = outputs.squeeze(1)
                    predicted = torch.round(outputs.data).clamp(0, 4).long()
                elif target == "OA":
                    probs = F.softmax(outputs, dim=1)
                    confidences, predicted = torch.max(probs, 1)
    
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
    
        val_bal_acc = balanced_accuracy_score(all_val_labels, all_val_preds)
        scheduler.step()
        
        if val_bal_acc > best_val_score:
            best_val_score = val_bal_acc
        trial.report(val_bal_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val_score

def run_optuna_search(mode: str = "BOTH", trials: int = 15, epochs: int = 50, target: str = "OA"):
    set_seed(42)

    logger = setup_logger(name="OptunaSearch", log_file="optuna.log")
    logger.info(f"Starting Hyperparameter Optimization {target} (Mode: {mode}) for {trials} trials")

    optuna_logger = logging.getLogger("optuna")
    optuna_logger.handlers.clear()
    for handler in logger.handlers:
        optuna_logger.addHandler(handler)

    study = optuna.create_study(
        study_name=f"model_hyperparams_optimization_{target}_{mode}",
        direction="maximize", # highest balanced accuracy
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    )
    bound_objective = lambda trial: objective(trial=trial, mode=mode, base_epochs=epochs, target=target)
    study.optimize(bound_objective, n_trials=trials, show_progress_bar=True)

    logger.info("-" * 15)
    logger.info(f"OPTIMIZATION COMPLETE {target} ({mode})")
    logger.info(f"Best Trial Number: {study.best_trial.number}")
    logger.info(f"Highest Balanced Validation Accuracy: {study.best_value:.2%}")

    logger.info("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")

    os.makedirs("./configs", exist_ok=True)
    best_config_data = {
        "target": target,
        "lateral_mode": mode,
        "lr": study.best_trial.params["lr"],
        "weight_decay": study.best_trial.params["weight_decay"],
        "eta_min": study.best_trial.params["eta_min"],
        "batch_size": study.best_trial.params["batch_size"],
        # "label_smoothing": study.best_trial.params["label_smoothing"],
    }

    output_path = Path(f"./configs/best_params_{target}_{mode}.json")
    with open(output_path, "w") as f:
        json.dump(best_config_data, f, indent=4)
    logger.info("=" * 15)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameters Optimization")
    parser.add_argument("--mode", type=str, default="BOTH", choices=["LEFT", "RIGHT", "BOTH"], help="Lateral anatomical standardization mode")
    parser.add_argument("--trials", type=int, default=25, help="Number of search trials")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs per trials")
    parser.add_argument("--target", type=str, default="KL", choices=["KL", "OA"], help="Target of the model (KL grading or Osteoarthritis separation)")
    args = parser.parse_args()
    run_optuna_search(mode=args.mode, trials=args.trials, epochs=args.epochs, target=args.target)