import argparse
import copy
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_handler import Fold_Handler, KLGradingDataset
from models import RCNN
from utils.logger import setup_logger
from utils.seed_setup import set_seed
from utils.transform import collate_fn, get_transform


def train_rcnn(timestamp=None):
    set_seed(42)
    logger = setup_logger(name="Knee Project", log_file="rcnn_training.log")

    timestamp = timestamp if timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Initializing RCNN training cross-validation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    root_dir = "./dataset"
    grade_path = "./dataset/KLGrade_label_with_5fold.xlsx"
    try:
        full_dataset = KLGradingDataset(
            root=root_dir,
            transforms=get_transform(),
            include_grades=True,
            grade_path=grade_path,
        )
        logger.info(f"Total dataset images mounted successfully: {len(full_dataset)}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    fold_handler = Fold_Handler(full_dataset)

    test_idx, cv_idx = [], []
    for i in range(len(full_dataset)):
        patient_id = full_dataset.patient_ids[i]
        assigned_fold = fold_handler.get_fold(patient_id)
        if assigned_fold == fold_handler.get_test_fold():
            test_idx.append(i)
        else:
            cv_idx.append(i)

    logger.info(f"Locked Test Set (Fold 0): {len(test_idx)} images")
    logger.info(f"Cross-Validation Pool: {len(cv_idx)} images")

    """
    Hyperparameters:
    - BATCH_SIZE: Stick with 2 to avoid OOM errors, and avoid 4 because GPU memory.
    """
    BATCH_SIZE = 2
    LR = 2e-4
    WEIGHT_DECAY = 1e-2
    NUM_EPOCHS = 30
    OPTIMIZER_NAME = "AdamW"

    cv_folds = fold_handler.get_cv_folds()
    for val_fold in cv_folds:
        logger.info(f"{'-' * 10}Starting Fold {val_fold}/{len(cv_folds)}{'-' * 10}")

        run_name = f"runs/rcnn_{timestamp}_fold{val_fold}"
        writer = SummaryWriter(log_dir=run_name)

        train_idx, val_idx = [], []
        for i in cv_idx:
            patient_id = full_dataset.patient_ids[i]
            if fold_handler.get_fold(patient_id) == val_fold:
                val_idx.append(i)
            else:
                train_idx.append(i)

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        logger.info(
            f"Fold {val_fold} | Training on: {len(train_subset)} images | Validating on: {len(val_subset)} images"
        )

        data_loader_train = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=3,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        data_loader_val = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=3,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # num_classes = 3 (0: Background, 1: Left Knee, 2: Right Knee)
        num_classes = 3
        model = RCNN(num_classes=num_classes)
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

        scheduler = CosineAnnealingLR(
            optimizer=optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
        )

        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0

            loop = tqdm(
                data_loader_train,
                desc=f"Fold {val_fold} - Epoch [{epoch + 1}/{NUM_EPOCHS}]",
            )
            # Training
            for images, targets in loop:
                # Move images and targets to GPU
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = torch.stack([loss for loss in loss_dict.values()]).sum()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                train_loss += losses.item()
                loop.set_postfix(loss=losses.item())

            avg_train_loss = train_loss / len(data_loader_train)

            # Validating
            # Model remains in train() mode for ensure return targets loss metrics cleanly
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in data_loader_val:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    loss_dict = model(images, targets)
                    losses = torch.stack([loss for loss in loss_dict.values()]).sum()
                    val_loss += losses.item()

            avg_val_loss = val_loss / len(data_loader_val)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                logger.info(
                    f"New best model found for Fold {val_fold} at epoch {epoch + 1} with validation loss: {best_val_loss:.4f}"
                )

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            logger.info(
                f"Epoch: {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr}"
            )

            writer.add_scalars(
                "Loss", {"Train": avg_train_loss, "Validation": avg_val_loss}, epoch + 1
            )
            writer.add_scalar("Learning_Rate", current_lr, epoch + 1)

        os.makedirs("./weights", exist_ok=True)
        save_path = Path("./weights/") / f"knee_rcnn_{timestamp}_fold{val_fold}.pth"

        if best_model_state is not None:
            torch.save(best_model_state, save_path)
            logger.info(
                f"Best model saved to {save_path} with validation loss: {best_val_loss:.4f}"
            )
        else:
            torch.save(model.state_dict(), save_path)
            logger.info(f"Final state saved directly to {save_path}")

        writer.add_hparams(
            hparam_dict={
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "weight_decay": WEIGHT_DECAY,
                "optimizer": OPTIMIZER_NAME,
                "epochs": NUM_EPOCHS,
            },
            metric_dict={"hparam/final_val_loss": avg_val_loss},
        )

        logger.info(f"Training complete! Fold {val_fold}")
        writer.close()

    logger.info("Training RCNN completed for all folds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RCNN model for knee grading")
    parser.add_argument("--timestamp", type=str, help="Timestamp for the whole process")
    args = parser.parse_args()
    train_rcnn(args.timestamp)
