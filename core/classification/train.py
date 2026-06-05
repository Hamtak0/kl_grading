import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # tell Matplotlib to use a headless, non-interactive backend called Agg (Anti-Grain Geometry)
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score

from utils.logger import setup_logger
from utils.metrics import create_confusion_matrix_figure
from utils.transform import get_transform, unnormalize_tensor
from utils.seed_setup import set_seed
from utils.config import load_toml_config
from core.data.dataset_handler import CachedKneeDataset, TransformWrapper, Fold_Handler
from core.classification.models import Classification_DenseNet

def train_classification(
        timestamp_override: str | None = None,
        config_path: str | Path | None = None,
        mode: str | None = None,
        target: str = "OA",
        loss_fn: str | None = "CE",
        strategy: str = "kfold_nested"
    ) -> tuple[Optional[str], Optional[str]]:
    set_seed(42) # Just same as kl grading seed cross-validation

    timestamp = timestamp_override if timestamp_override else datetime.now().strftime('%Y%m%d_%H%M%S')

    # default parameters from config.toml, can be overridden by JSON config or command-line args
    env_cfg = load_toml_config(Path("configs/config.toml"))
    MODE = mode.upper() if mode else env_cfg["experiment"]["lateral_mode"] # "LEFT", "RIGHT", or "BOTH"
    BATCH_SIZE = env_cfg["classification_defaults"]["batch_size"]
    NUM_EPOCHS = env_cfg["classification_defaults"]["num_epochs"]
    LR = 1.4768e-4
    WEIGHT_DECAY = 9.2914e-5
    ETA_MIN = 1.64e-6
    # LABEL_SMOOTHING = 0.25422

    results_dir = Path(f"./results/classification/{timestamp}_{strategy}_{target}_{MODE}_{loss_fn}")
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "weights").mkdir(exist_ok=True)
    (results_dir / "runs").mkdir(exist_ok=True)
    (results_dir / "metrics" / "confusion_matrices").mkdir(parents=True, exist_ok=True)

    logger = setup_logger(name="Classification Training", log_file=str(results_dir / "densenet_training.log"))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if config_path and Path(config_path).exists():
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
    else:
        logger.warning(f"No valid config file found at {config_path}. Falling back to baseline TOML defaults.")

    # To handle nvidia dedicated GPU memory
    physical_batch_size = 2
    accumulation_steps = max(1, BATCH_SIZE // physical_batch_size)

    logger.info(f"Training parameters {target} -> Mode: {MODE}, Batch: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}, LR: {LR:.6f}")

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
    all_folds = fold_handler.get_all_folds()
    test_vault = fold_handler.get_test_fold()
    cv_folds = fold_handler.get_cv_folds()

    fold_scores = []
    gpu_train_transform = get_transform(rotation=0, jitter=0.3, sharpness=2, is_train=True).to(device)
    gpu_val_transform = get_transform(rotation=0, jitter=0, sharpness=0, is_train=False).to(device)

    run_configs = []
    if strategy == "single_holdout":
        #* Train 4 models. Test fold is simply held out and val fold shifts to the next fold (60/20/20).
        for v_fold in cv_folds:
            t_folds = [f for f in cv_folds if f != v_fold]
            run_configs.append({"model_id": v_fold, "test_fold": test_vault, "val_fold": v_fold, "train_folds": t_folds})
    elif strategy == "kfold_blind":
        #* Train 5 models. Test shifts and no validation set (80/20).
        for te_fold in all_folds:
            tr_folds = [f for f in all_folds if f != te_fold]
            run_configs.append({"model_id": te_fold, "test_fold": te_fold, "val_fold": None, "train_folds": tr_folds})
    elif strategy == "kfold_nested":
        #* Train 5 models. Test shifts and val shifts (60/20/20). Okada sensei's approach.
        for te_fold in all_folds:
            v_fold = (te_fold + 1) % len(all_folds)
            tr_folds = [f for f in all_folds if f not in (te_fold, v_fold)]
            run_configs.append({"model_id": te_fold, "test_fold": te_fold, "val_fold": v_fold, "train_folds": tr_folds})

    logger.info(f"-- Executing with strategy: {strategy.lower()}")

    # --- --- --- ---
    # Training Loop
    # --- --- --- ---
    for config in run_configs:
        model_id = config["model_id"]
        test_fold = config["test_fold"]
        val_fold = config["val_fold"]
        train_folds = config["train_folds"]

        logger.info(f"Model {model_id} | Train: {train_folds} | Val: {val_fold} | Test: {test_fold}")

        run_name = results_dir / "runs" / f"fold{model_id}"
        writer = SummaryWriter(log_dir=run_name)

        train_idx = []
        val_idx = []
        for i, sample in enumerate(cv_classifier_dataset.samples):
            base_id = sample["patient_id"].split("_")[0] # ID001, ID002, etc.
            assigned_fold = fold_handler.get_fold(base_id)
            if assigned_fold in train_folds:
                train_idx.append(i)
            elif assigned_fold == val_fold:
                val_idx.append(i)
    
        train_subset = Subset(cv_classifier_dataset, train_idx)
        train_dataset = TransformWrapper(train_subset, mode=MODE)
        data_loader_train = DataLoader(train_dataset, batch_size=physical_batch_size, shuffle=True, num_workers=3, pin_memory=True)

        if val_fold is not None:
            val_subset = Subset(cv_classifier_dataset, val_idx)
            val_dataset = TransformWrapper(val_subset, mode=MODE)
            data_loader_val = DataLoader(val_dataset, batch_size=physical_batch_size, shuffle=False, num_workers=3, pin_memory=True)

        if target == "KL": # 5 grades
            if loss_fn == "MSE": num_classes = 1
            elif loss_fn == "CE": num_classes = 5
            model = Classification_DenseNet(num_classes=num_classes).to(device)
            train_labels = [cv_classifier_dataset.samples[i]['kl_grade'] for i in train_idx]
            raw_weights = compute_class_weight(class_weight='balanced', classes=np.arange(5), y=train_labels)
            weights_tensor = torch.tensor(raw_weights, dtype=torch.float32).to(device)
            if loss_fn == "MSE": criterion = nn.MSELoss(reduction='none')
            elif loss_fn == "CE": criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        elif target == "OA": # 2 grades
            model = Classification_DenseNet(num_classes=2).to(device)
            train_labels = [cv_classifier_dataset.samples[i]['oa_grade'] for i in train_idx]
            raw_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=train_labels)
            weights_tensor = torch.tensor(raw_weights, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN)

        best_val_score = 0.0
        best_epoch = 0
        scaler = torch.amp.GradScaler('cuda')

        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss, correct_train, total_train = 0.0, 0, 0
            optimizer.zero_grad()

            loop = tqdm(data_loader_train, desc=f"Model {model_id} - Epoch [{epoch+1}/{NUM_EPOCHS}]")

            for step, (images, kl, oa, _) in enumerate(loop):
                # print("Labels in current batch:", labels)
                if target == "KL": labels = kl.to(device)
                elif target == "OA": labels = oa.to(device)
                images = gpu_train_transform(images.to(device))
    
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    if target == "KL" and loss_fn == "MSE":
                        outputs = outputs.squeeze(1)
                        unweighted_loss = criterion(outputs, labels.float())
                        weighted_sum = (unweighted_loss * weights_tensor[labels.long()]).sum()
                        raw_loss = weighted_sum / weights_tensor[labels.long()].sum()
                        predicted = torch.round(outputs.data).clamp(0, 4).long()
                    elif target == "OA" or loss_fn == "CE":
                        raw_loss = criterion(outputs, labels)
                        _, predicted = torch.max(outputs.data, 1)

                    loss = raw_loss / accumulation_steps

                scaler.scale(loss).backward() # add current gradient to the buffer
                # loss.backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(data_loader_train):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    # optimizer.step()

                train_loss += raw_loss.item() * images.size(0)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                loop.set_postfix(loss=raw_loss.item())

            avg_train_loss = train_loss / total_train 
            train_acc = correct_train / total_train
            scheduler.step()

            if val_fold is not None:
                model.eval()
                val_loss, correct_val, total_val = 0.0, 0, 0
                all_val_labels = []
                all_val_preds = []
                epoch_val_images = []

                with torch.no_grad():
                    for images, kl, oa, patient_ids in data_loader_val:
                        if target == "KL": labels = kl.to(device)
                        elif target == "OA": labels = oa.to(device)
                        images = gpu_val_transform(images.to(device))
    
                        with torch.amp.autocast('cuda'):
                            outputs = model(images)
                            if target == "KL" and loss_fn == "MSE":
                                outputs_mse = outputs.squeeze(1)
                                unweighted_loss = criterion(outputs_mse, labels.float())
                                weighted_sum = (unweighted_loss * weights_tensor[labels.long()]).sum()
                                loss_val = weighted_sum / weights_tensor[labels.long()].sum()
                            elif target == "OA" or loss_fn == "CE":
                                loss_val = criterion(outputs, labels)

                        outputs = outputs.float()
                        if target == "KL" and loss_fn == "MSE":
                            outputs_mse = outputs.squeeze(1)
                            predicted = torch.round(outputs_mse).clamp(0, 4).long()
                            distances = torch.abs(outputs_mse - predicted.float()) 
                            confidences = torch.clamp(1.0 - distances, min=0.0, max=1.0)
                        elif target == "OA" or loss_fn == "CE":
                            probs = F.softmax(outputs, dim=1)
                            confidences, predicted = torch.max(probs, 1)

                        val_loss += loss_val.item() * images.size(0)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()

                        all_val_labels.extend(labels.cpu().numpy())
                        all_val_preds.extend(predicted.cpu().numpy())

                        # unnormalize and store images for confusion matrix visualization
                        images_np = unnormalize_tensor(images)
                        for i in range(len(labels)):
                            epoch_val_images.append({
                                "image": images_np[i],
                                "true_label": labels[i].item(),
                                "pred_label": predicted[i].item(),
                                "confidence": confidences[i].item(),
                                "patient_id": patient_ids[i] 
                            })

                avg_val_loss = val_loss / total_val
                val_acc = correct_val / total_val
                val_balanced_acc = balanced_accuracy_score(all_val_labels, all_val_preds)

                logger.info(f"Model {model_id} | Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.2%}) | Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.2%}) | Bal Acc: {val_balanced_acc:.2%}")

                cm_title = f'{target} Mode {MODE} Model {model_id} Epoch {epoch+1}\nVal Acc: {val_acc:.2%} | Bal Acc: {val_balanced_acc:.2%}'
                fig = create_confusion_matrix_figure(all_val_labels, all_val_preds, target=target, title=cm_title)

                # TensorBoard Logging
                writer.add_scalars("Loss", {"Train": avg_train_loss, "Validation": avg_val_loss}, epoch+1)
                writer.add_scalars("Accuracy", {"Train": train_acc, "Validation": val_acc, "Balanced": val_balanced_acc}, epoch+1)

                # Early Stopping
                if val_balanced_acc > best_val_score:
                    best_val_score = val_balanced_acc
                    best_epoch = epoch + 1

                    torch.save(model.state_dict(), results_dir / "weights" / f"fold{model_id}.pth")

                    fig.savefig(results_dir / "metrics" / "validation_confusion_matrices" / f"cm_fold{model_id}.png", dpi=150)

                    img_save_dir = results_dir / "images_val" / f"fold{model_id}"
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

                    logger.info(f"New best model saved for model {model_id} at epoch {epoch+1} with Balanced accuracy: {best_val_score:.2%}")
            
                plt.close(fig)

            # If no validation fold.
            else:
                logger.info(f"Model {model_id} | Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.2%})")
                writer.add_scalars("Loss", {"Train": avg_train_loss}, epoch+1)
                writer.add_scalars("Accuracy", {"Train": train_acc}, epoch+1)

                if epoch == NUM_EPOCHS - 1: # Save the final model
                    torch.save(model.state_dict(), results_dir / "weights" / f"fold{model_id}.pth")
                    logger.info(f"Blind training complete. Model {model_id} at epoch {epoch+1}")

        writer.close()

        if val_fold is not None:
            fold_scores.append(best_val_score)
            logger.info(f"Model {model_id} complete! Best validation score: {best_val_score:.4f} at epoch: {best_epoch}")
        else:
            logger.info(f"Model {model_id} complete! With blind training (no validation)")
    
    if len(fold_scores) > 0:
        logger.info(f"Average Validation Score across runs ({target}): {np.mean(fold_scores):.4%}")
        
    logger.info(f"Training run completed {target} {loss_fn} {MODE}. Timestamp: {timestamp}")
    logger.info("=" * 20)

    return timestamp, MODE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Classification")
    parser.add_argument("-c", "--config", type=str, help="Path to JSON hyperparameters config file")
    parser.add_argument("-m", "--mode", type=str, choices=["LEFT", "RIGHT", "BOTH"], help="Override training mode")
    parser.add_argument("-ts", "--timestamp", type=str, help="Timestamp of the whole process")
    parser.add_argument("-tg", "--target", type=str, choices=["KL", "OA"], help="Target of the model (KL grading or Osteoarthritis separation)")
    parser.add_argument("-l", "--loss", type=str, choices=["CE", "MSE"], help="Loss function to be trained with")
    parser.add_argument("-s", "--strategy", type=str, choices=["single_holdout", "kfold_blind", "kfold_nested"], help="Cross-validation architecture to use")
    args = parser.parse_args()
    train_classification(config_path=args.config, mode_override=args.mode, timestamp_override=args.timestamp, target=args.target, loss_fn=args.loss, strategy=args.strategy)