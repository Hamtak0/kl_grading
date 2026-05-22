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

def objective(trial, mode="BOTH", base_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameter Search Space
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    eta_min = trial.suggest_float("eta_min", 1e-7, 1e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    # label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.3)
    
    env_cfg = load_toml_config("configs/config.toml")
    cv_classifier_dataset = CachedKneeDataset(
        cache_dir=env_cfg["dataset"]["cache_dir"],
        root=env_cfg["dataset"]["root_dir"],
        grade_path=env_cfg["dataset"]["excel_file"]
    )
    fold_handler = Fold_Handler(cv_classifier_dataset)
    cv_folds = fold_handler.get_cv_folds()

    gpu_train_transform = get_transform(rotation=10, jitter=0.3, sharpness=2, is_train=True).to(device)
    gpu_val_transform = get_transform(rotation=0, jitter=0, sharpness=0, is_train=False).to(device)
    # grade_values = torch.arange(5, dtype=torch.float32).to(device)

    fold_val_scores_per_epoch = np.zeros((len(cv_folds), base_epochs)) # shape: cv_folds x base_epochs

    for f_idx, val_fold in enumerate(cv_folds):
        train_idx, val_idx = [], []

        for i, sample in enumerate(cv_classifier_dataset.samples):
            base_id = sample["patient_id"].split("_")[0]
            assigned_fold = fold_handler.get_fold(base_id)
            if assigned_fold == fold_handler.get_test_fold():
                continue
            elif assigned_fold == val_fold:
                val_idx.append(i)
            else:
                train_idx.append(i)

        train_subset = Subset(cv_classifier_dataset, train_idx)
        val_subset = Subset(cv_classifier_dataset, val_idx)

        train_dataset = TransformWrapper(train_subset, mode=mode)
        val_dataset = TransformWrapper(val_subset, mode=mode)

        data_loader_train = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True,
        )
        data_loader_val = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=3,
            pin_memory=True,
        )

        # model = Classification_ResNet(num_classes=5).to(device)
        model = Classification_DenseNet(num_classes=1).to(device)

        train_labels = [cv_classifier_dataset.samples[i]["kl_grade"] for i in train_idx]
        class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(5), y=train_labels)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
        # criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)
        criterion = nn.MSELoss(reduction='none')
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=base_epochs, eta_min=eta_min)

        # scaler = torch.amp.GradScaler('cuda')

        for epoch in range(base_epochs):
            model.train()
            for images, labels, oa, _ in data_loader_train: # images, labels, oa, pid
                images, labels = images.to(device), labels.to(device)
                images = gpu_train_transform(images)

                optimizer.zero_grad()

                # with torch.amp.autocast('cuda'):
                outputs = model(images).squeeze(1)
                batch_weights = weights_tensor[labels.long()]
                unweighted_loss = criterion(outputs, labels.float())
                loss = (unweighted_loss * batch_weights).mean()
                    # outputs = model(images)
                    # loss = criterion(outputs, labels)
                
                # scaler.scale(loss).backward()
                loss.backward()

                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()

            model.eval()
            all_val_labels = []
            all_val_preds = []

            with torch.no_grad():
                for images, labels, oa, _ in data_loader_val:
                    images, labels = images.to(device), labels.to(device)
                    images = gpu_val_transform(images)

                    # outputs = model(images)
                    outputs = model(images).squeeze(1)
                    predicted = torch.round(outputs.data).clamp(0, 4).long()

                    # probs = F.softmax(outputs, dim=1)
                    # _, predicted = torch.max(probs, 1)

                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_preds.extend(predicted.cpu().numpy())

            val_bal_acc = balanced_accuracy_score(all_val_labels, all_val_preds)
            fold_val_scores_per_epoch[f_idx, epoch] = val_bal_acc
            scheduler.step()

            current_step = (f_idx * base_epochs) + epoch
            trial.report(val_bal_acc, current_step)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    mean_scores_per_epoch = np.mean(fold_val_scores_per_epoch, axis=0)
    best_averaged_score = float(np.max(mean_scores_per_epoch))

    return best_averaged_score

#TODO: Because right now the pipeline is using command line instead of calling the function so maybe just maybe put this in like function name and then called from the pipeline
#TODO: to also pass through the timestamp and maybe compare the hyperparameters from json with timestamp in its name.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameters Optimization")
    parser.add_argument("--mode", type=str, default="BOTH", choices=["LEFT", "RIGHT", "BOTH"], help="Lateral anatomical standardization mode")
    parser.add_argument("--trials", type=int, default=25, help="Number of search trials")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs per trials")
    args = parser.parse_args()

    set_seed(42)

    logger = setup_logger(name="OptunaSearch", log_file="optuna.log")
    logger.info(f"Starting Hyperparameter Optimization (Mode: {args.mode}) for {args.trials} trials...")

    optuna_logger = logging.getLogger("optuna")
    optuna_logger.handlers.clear()
    for handler in logger.handlers:
        optuna_logger.addHandler(handler)

    study = optuna.create_study(
        study_name=f"model_hyperparams_optimization_{args.mode}",
        direction="maximize", # highest balanced accuracy
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    )
    
    bound_objective = lambda trial: objective(trial, mode=args.mode, base_epochs=args.epochs)

    study.optimize(bound_objective, n_trials=args.trials, show_progress_bar=True)

    logger.info("-" * 15)
    logger.info(f"OPTIMIZATION COMPLETE ({args.mode})")
    logger.info(f"Best Trial Number: {study.best_trial.number}")
    logger.info(f"Highest Balanced Validation Accuracy: {study.best_value:.2%}")

    logger.info("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")

    os.makedirs("./configs", exist_ok=True)
    best_config_data = {
        "lateral_mode": args.mode,
        "lr": study.best_trial.params["lr"],
        "weight_decay": study.best_trial.params["weight_decay"],
        "eta_min": study.best_trial.params["eta_min"],
        "batch_size": study.best_trial.params["batch_size"],
        # "label_smoothing": study.best_trial.params["label_smoothing"],
    }

    output_path = f"./configs/best_params_{args.mode}.json"
    with open(output_path, "w") as f:
        json.dump(best_config_data, f, indent=4)

    logger.info(f"Success! Optimal parameters safely exported to {output_path}")
    logger.info("=" * 15)
