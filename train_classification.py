import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # tell Matplotlib to use a headless, non-interactive backend called Agg (Anti-Grain Geometry)
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score

from utils.logger import setup_logger
from utils.metrics import create_confusion_matrix_figure
from utils.transform import get_transform, unnormalize_tensor
from utils.seed_setup import set_seed
from utils.config import load_toml_config
from dataset_handler import CachedKneeDataset, TransformWrapper, Fold_Handler
from models import Classification_ResNet, Classification_DenseNet

def train_classification(config_path=None, mode_override=None, timestamp_override=None) -> Tuple[Optional[str], Optional[str]]:
    set_seed(42) # Just same as kl grading seed cross-validation

    logger = setup_logger(name="KneeClassifier", log_file="classifier_training.log")
    timestamp = timestamp_override if timestamp_override else datetime.now().strftime('%Y%m%d_%H%M%S')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    env_cfg = load_toml_config("configs/config.toml")
    MODE = env_cfg["experiment"]["lateral_mode"] # "LEFT", "RIGHT", or "BOTH"
    BATCH_SIZE = env_cfg["training_defaults"]["batch_size"]
    NUM_EPOCHS = env_cfg["training_defaults"]["num_epochs"]
    LR = 1.4768e-4
    WEIGHT_DECAY = 9.2914e-5
    ETA_MIN = 1.64e-6
    # LABEL_SMOOTHING = 0.25422

    if config_path and os.path.exists(config_path):
        logger.info(f"Loading parameters from {config_path}")
        with open(config_path, "r") as f:
            hyper_cfg = json.load(f)
        MODE = hyper_cfg.get("lateral_mode", MODE)
        BATCH_SIZE = hyper_cfg.get("batch_size", BATCH_SIZE)
        NUM_EPOCHS = hyper_cfg.get("epochs", NUM_EPOCHS)
        LR = hyper_cfg.get("lr", LR)
        WEIGHT_DECAY = hyper_cfg.get("weight_decay", WEIGHT_DECAY)
        ETA_MIN = hyper_cfg.get("eta_min", ETA_MIN)
        # LABEL_SMOOTHING = hyper_cfg.get("label_smoothing", LABEL_SMOOTHING)
    
    if mode_override:
        MODE = mode_override.upper()

    logger.info(f"Training parameters -> Mode: {MODE}, Batch: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}, LR: {LR:.6f}")

    try:
        cv_classifier_dataset = CachedKneeDataset(
            cache_dir=env_cfg["dataset"]["cache_dir"],
            root=env_cfg["dataset"]["root_dir"],
            grade_path=env_cfg["dataset"]["excel_file"]
        )
        logger.info(f"Total knees extracted: {len(cv_classifier_dataset)}")
    except Exception as e:
        logger.error(f"Failed to load the dataset: {e}")
        return None, None

    fold_handler = Fold_Handler(cv_classifier_dataset)

    test_idx = []
    cv_idx = []

    for i, sample in enumerate(cv_classifier_dataset.samples):
        base_id = sample["patient_id"].split("_")[0]
        assigned_fold = fold_handler.get_fold(base_id)

        if assigned_fold == fold_handler.get_test_fold():
            test_idx.append(i) # Fold 0
        else:
            cv_idx.append(i) # Folds 1-4

    logger.info(f"Locked Test Set (Fold 0): {len(test_idx)} knees")
    logger.info(f"Available for Cross-Validation: {len(cv_idx)} knees")

    cv_folds = fold_handler.get_cv_folds()

    #* Notes: We have 5 folds in total
    #* Fold 0 is reserved as a locked test set that we never touch during training or validation.
    #* Perform 4-fold cross-validation and use 1 fold for validation and the remaining 3 folds for training in each iteration.
    #* Therefore, we use 60% of the data for training, 20% for validation, and 20% for testing.
    for val_fold in cv_folds:
        logger.info(f"{'-'*10}Starting Fold {val_fold}/{len(cv_folds)}{'-'*10}")

        run_name = f"runs/classification_{timestamp}_fold{val_fold}_{MODE}"
        writer = SummaryWriter(log_dir=run_name)

        train_idx, val_idx = [], []

        for i in cv_idx:
            sample = cv_classifier_dataset.samples[i]
            base_id = sample["patient_id"].split("_")[0] # ID001, ID002, etc.
            assigned_fold = fold_handler.get_fold(base_id)

            # Use another one fold for validation, the rest for training
            if assigned_fold == val_fold:
                val_idx.append(i)
            else:
                train_idx.append(i)
    
        train_subset = Subset(cv_classifier_dataset, train_idx)
        val_subset = Subset(cv_classifier_dataset, val_idx)

        train_dataset = TransformWrapper(
            train_subset,
            # get_classification_transform(rotation=5, jitter=0.05, is_train=True),
            mode=MODE
        )
        val_dataset = TransformWrapper(
            val_subset,
            # get_classification_transform(rotation=0, jitter=0, is_train=False),
            mode=MODE
        )

        data_loader_train = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=3,
            pin_memory=True,
            prefetch_factor=2,
        )
        data_loader_val = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=3,
            pin_memory=True,
        )

        train_labels = [cv_classifier_dataset.samples[i]['kl_grade'] for i in train_idx]
        raw_weights = compute_class_weight(class_weight='balanced', classes=np.arange(5), y=train_labels)
        weights_tensor = torch.tensor(raw_weights, dtype=torch.float32).to(device)

        # model = Classification_CNN(num_classes=5, in_channels=1).to(device)
        # model = Classification_ResNet(num_classes=5).to(device)
        model = Classification_DenseNet(num_classes=1).to(device)
        # criterion = nn.CrossEntropyLoss(
        #     weight=weights_tensor,
        #     label_smoothing=LABEL_SMOOTHING
        # )
        criterion = nn.MSELoss(reduction='none')
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN)

        best_val_score = 0.0
        best_epoch = 0
        gpu_train_transform = get_transform(rotation=10, jitter=0.3, sharpness=2, is_train=True).to(device)
        gpu_val_transform = get_transform(rotation=0, jitter=0, sharpness=0, is_train=False).to(device)

        # scaler = torch.amp.GradScaler('cuda')

        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            loop = tqdm(data_loader_train, desc=f"Fold {val_fold} - Epoch [{epoch+1}/{NUM_EPOCHS}]")

            for images, labels, oa, _ in loop:
                # print("Labels in current batch:", labels)
                images = images.to(device)
                labels = labels.to(device)

                images = gpu_train_transform(images)

                optimizer.zero_grad()

                outputs = model(images).squeeze(1)
                batch_weights = weights_tensor[labels.long()]
                unweighted_loss = criterion(outputs, labels.float())
                loss = (unweighted_loss * batch_weights).mean()
                # with torch.amp.autocast('cuda'):
                    # outputs = model(images)
                    # loss = criterion(outputs, labels)
                    # probs = F.softmax(outputs, dim=1)

                # scaler.scale(loss).backward()
                loss.backward()

                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

                # Calculate Training Accuracy
                # _, predicted = torch.max(outputs.data, 1)
                predicted = torch.round(outputs.data).clamp(0, 4).long()

                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                loop.set_postfix(loss=loss.item())

            avg_train_loss = train_loss / len(train_subset)
            train_acc = correct_train / total_train

            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            all_val_labels = []
            all_val_preds = []

            epoch_val_images = []

            with torch.no_grad():
                for images, labels, oa, patient_ids in data_loader_val:
                    images = images.to(device)
                    labels = labels.to(device)

                    images = gpu_val_transform(images)

                    # outputs = model(images)
                    outputs = model(images).squeeze(1)
                    batch_weights = weights_tensor[labels.long()]
                    unweighted_loss_val = criterion(outputs, labels.float())
                    # loss_val = criterion(outputs, labels)
                    loss_val = (unweighted_loss_val * batch_weights).mean()
                    # probs_val = F.softmax(outputs, dim=1)

                    val_loss += loss_val.item() * images.size(0)

                    # confidences, predicted = torch.max(probs_val, 1)
                    predicted = torch.round(outputs.data).clamp(0, 4).long()

                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_preds.extend(predicted.cpu().numpy())

                    # unnormalize and store images for confusion matrix visualization
                    images_np = unnormalize_tensor(images)

                    distances = torch.abs(outputs.data - predicted.float()) 
                    confidences = torch.clamp(1.0 - distances, min=0.0, max=1.0)

                    for i in range(len(labels)):
                        epoch_val_images.append({
                            "image": images_np[i],
                            "true_label": labels[i].item(),
                            "pred_label": predicted[i].item(),
                            "confidence": confidences[i].item(),
                            "patient_id": patient_ids[i] 
                        })

            avg_val_loss = val_loss / len(val_subset)
            val_acc = correct_val / total_val
            val_balanced_acc = balanced_accuracy_score(all_val_labels, all_val_preds)

            scheduler.step()

            logger.info(f"Fold {val_fold} | Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.2%}) | Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.2%}) | Bal Acc: {val_balanced_acc:.2%}")

            cm_title = f'Mode {MODE} Fold {val_fold} Epoch {epoch+1}\nVal Acc: {val_acc:.2%} | Bal Acc: {val_balanced_acc:.2%}'
            fig = create_confusion_matrix_figure(all_val_labels, all_val_preds, num_classes=5, title=cm_title)

            # TensorBoard Logging
            writer.add_scalars("Loss", {"Train": avg_train_loss, "Validation": avg_val_loss}, epoch+1)
            writer.add_scalars("Accuracy", {"Train": train_acc, "Validation": val_acc, "Balanced": val_balanced_acc}, epoch+1)
            writer.add_figure("Confusion Matrix", fig, epoch+1)

            # Early Stopping
            if val_balanced_acc > best_val_score:
                best_val_score = val_balanced_acc
                best_epoch = epoch+1

                os.makedirs("./weights", exist_ok=True)
                torch.save(model.state_dict(), f"./weights/knee_class_{timestamp}_fold{val_fold}_{MODE}.pth")

                os.makedirs("./confusion_matrices", exist_ok=True)
                fig.savefig(f"./confusion_matrices/cm_{timestamp}_fold{val_fold}_{MODE}.png", dpi=150)

                img_save_dir = Path(f"./images_val/{timestamp}_fold{val_fold}_{MODE}")
                if img_save_dir.exists():
                    shutil.rmtree(img_save_dir)
                img_save_dir.mkdir(parents=True, exist_ok=True)

                for _, item in enumerate(epoch_val_images):
                    fig_img, ax_img = plt.subplots(figsize=(4, 4))
                    ax_img.imshow(item["image"][:, :, 0], cmap="gray")

                    title_color = "green" if item["true_label"] == item["pred_label"] else "red"
                    ax_img.set_title(
                        f'ID: {item["patient_id"]}\nTrue: {item["true_label"]} | Pred: {item["pred_label"]}\nConf: {item["confidence"]:.1%}',
                        color=title_color,
                        fontweight='bold',
                        fontsize=8
                    )
                    ax_img.axis('off')

                    filename = img_save_dir / f"{item['patient_id']}_T{item['true_label']}_P{item['pred_label']}.png"
                    fig_img.savefig(filename, bbox_inches='tight')
                    plt.close(fig_img)

                logger.info(f"New best model saved for Fold {val_fold} at epoch {epoch+1} with Balanced accuracy: {best_val_score:.2%}")
            
            plt.close(fig)

        writer.close()
        logger.info(f"Fold {val_fold} complete! Best validation score: {best_val_score:.4f} at epoch: {best_epoch}")
    
    logger.info(f"Training complete! timestamp: {timestamp}")
    logger.info("=" * 20)

    return timestamp, MODE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Classification")
    parser.add_argument("--config", type=str, help="Path to JSON hyperparameters config file")
    parser.add_argument("--mode", type=str, choices=["LEFT", "RIGHT", "BOTH"], help="Override training mode")
    parser.add_argument("--timestamp", type=str, help="Timestamp of the whole process")
    args = parser.parse_args

    train_classification(config_path=args.config, mode_override=args.mode, timestamp_override=args.timestamp)