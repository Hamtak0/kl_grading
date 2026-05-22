import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score
from tqdm import tqdm

from utils.seed_setup import set_seed
from utils.logger import setup_logger
from utils.metrics import create_confusion_matrix_figure
from utils.transform import get_transform, unnormalize_tensor
from utils.config import load_toml_config
from dataset_handler import CachedKneeDataset, TransformWrapper, Fold_Handler
from models import Classification_ResNet, Classification_DenseNet

def test_classification(timestamp_override=None, mode_override="BOTH"):
    set_seed(42)

    logger = setup_logger(name="KneeTest", log_file="classifier_testing.log")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Starting Testing on device: {device}")

    timestamp = timestamp_override if timestamp_override else "20260511_154935"
    MODE = mode_override.upper()
    logger.info(f"Ensemble timestamp: {timestamp}, mode: {MODE}")

    env_cfg = load_toml_config("configs/config.toml")
    try:
        cv_classifier_dataset = CachedKneeDataset(
            cache_dir=env_cfg["dataset"]["cache_dir"],
            root=env_cfg["dataset"]["root_dir"],
            grade_path=env_cfg["dataset"]["excel_file"]
        )
        logger.info(f"Successfully loaded CachedKneeDataset total size: {len(cv_classifier_dataset)}.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    fold_handler = Fold_Handler(cv_classifier_dataset)

    test_idx = []
    for i, sample in enumerate(cv_classifier_dataset.samples):
        base_id = sample["patient_id"].split("_")[0]
        assigned_fold = fold_handler.get_fold(base_id)
        if assigned_fold == fold_handler.get_test_fold():
            test_idx.append(i)
    logger.info(f"Extracted {len(test_idx)} test indices from fold 0.")

    test_subset = Subset(cv_classifier_dataset, test_idx)
    test_dataset = TransformWrapper(test_subset, mode=MODE)

    data_loader_test = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=3,
        pin_memory=True
    )

    models = []
    for fold in fold_handler.get_cv_folds():
        # model = Classification_ResNet(num_classes=5, freeze_backbone=True).to(device)
        model = Classification_DenseNet(num_classes=1).to(device)
        weight_path = Path(f"./weights/knee_class_{timestamp}_fold{fold}_{MODE}.pth")
        
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            model.eval() 
            models.append(model)
            logger.info(f"Successfully loaded Fold {fold}_{MODE} weights.")
        except Exception as e:
            logger.error(f"Could not load {weight_path}. Error: {e}")
            return
    if not models:
        logger.error("No models were loaded. Aborting testing.")
        return

    all_labels = []
    ensemble_preds = []
    ensemble_confs = []
    parent_records = []
    test_images_data = []

    gpu_test_transform = get_transform(rotation=0, jitter=0, sharpness=0, is_train=False).to(device)
    
    logger.info(f"Running ground truth test data through the {len(models)}-model ensemble...")

    with torch.no_grad():
        for images, labels, pids in tqdm(data_loader_test, desc=f"Testing fold 0 ({MODE})"):
            images = images.to(device)
            labels = labels.to(device)

            images = gpu_test_transform(images)
            
            all_labels.extend(labels.cpu().numpy())

            # ensemble_outputs = torch.zeros((images.size(0), 5)).to(device)
            ensemble_outputs = torch.zeros((images.size(0),)).to(device)

            for model in models:
                # outputs = model(images)
                # probabilities = F.softmax(outputs, dim=1)
                # ensemble_outputs += probabilities
                outputs = model(images).squeeze(1)
                ensemble_outputs += outputs
                
            # avg_probabilities = ensemble_outputs / len(models)
            # confidences, predicted = torch.max(avg_probabilities.data, 1)
            avg_outputs = ensemble_outputs / len(models)
            predicted = torch.round(avg_outputs).clamp(0, 4).long()

            distances = torch.abs(avg_outputs - predicted.float())
            confidences = torch.clamp(1.0 - distances, min=0.0, max=1.0)
            
            ensemble_preds.extend(predicted.cpu().numpy())
            ensemble_confs.extend(confidences.cpu().numpy())
            parent_records.extend(pids)

            images_np = unnormalize_tensor(images.cpu())
            for i in range(len(labels)):
                test_images_data.append({
                    "image": images_np[i],
                    "true_label": labels[i].item(),
                    "pred_label": predicted[i].item(),
                    "confidence": confidences[i].item(),
                    "patient_id": pids[i]
                })

    # Evaluate and Log Results
    all_labels = np.array(all_labels)
    ensemble_preds = np.array(ensemble_preds)
    ensemble_confs = np.array(ensemble_confs)
    parent_records = np.array(parent_records)

    test_acc = accuracy_score(all_labels, ensemble_preds)
    test_bal_acc = balanced_accuracy_score(all_labels, ensemble_preds)

    logger.info(f"== Final ensemble test accuracy (Overall): {test_acc:.2%}")
    logger.info(f"== Final balanced accuracy (Overall): {test_bal_acc:.2%}")

    logger.info("Overall Ensemble Classification Report:")
    report = classification_report(all_labels, ensemble_preds, target_names=[f"Grade {i}" for i in range(5)], zero_division=0)
    logger.info("\n" + report)

    # creates results directory for grouping test together
    os.makedirs("./results", exist_ok=True)
    results_dir = Path(f"./results/{timestamp}_{MODE}")
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Consolidating all performance into: {results_dir}")

    logger.info(f"Bilateral subgroup analysis (Standardization: {MODE}):")
    left_mask = np.array([pid.endswith("_L") for pid in parent_records])
    right_mask = np.array([pid.endswith("_R") for pid in parent_records])

    if np.any(left_mask):
        l_labels = all_labels[left_mask]
        l_preds = ensemble_preds[left_mask]
        l_acc = accuracy_score(l_labels, l_preds)
        l_bal_acc = balanced_accuracy_score(l_labels, l_preds)
        logger.info(f"Left knees (count: {np.sum(left_mask)}) - Accuracy: {l_acc:.2%}, Balanced Accuracy: {l_bal_acc:.2%}")

        cm_title_l = f"Left knees ({MODE})\nAccuracy: {l_acc:.2f}, Balance Accurary: {l_bal_acc:.2f}"
        fig_l = create_confusion_matrix_figure(l_labels, l_preds, num_classes=5, title=cm_title_l)
        fig_l.savefig(results_dir / f"cm_LEFT.png", dpi=300)
        plt.close(fig_l)
    else:
        logger.warning("No left knees found in the test set")

    if np.any(right_mask):
        r_labels = all_labels[right_mask]
        r_preds = ensemble_preds[right_mask]
        r_acc = accuracy_score(r_labels, r_preds)
        r_bal_acc = balanced_accuracy_score(r_labels, r_preds)
        logger.info(f"Right knees (count: {np.sum(right_mask)}) - Accuracy: {r_acc:.2%}, Balanced Accuracy: {r_bal_acc:.2%}")

        cm_title_r = f"Left knees ({MODE})\nAccuracy: {r_acc:.2f}, Balance Accurary: {r_bal_acc:.2f}"
        fig_r = create_confusion_matrix_figure(r_labels, r_preds, num_classes=5, title=cm_title_r)
        fig_r.savefig(results_dir / f"cm_RIGHT.png", dpi=300)
        plt.close(fig_r)
    else:
        logger.warning("No right knees found in the test set")

    results_df = pd.DataFrame({
        'Patient Knee': parent_records,
        'Side': [
            "L" if pid.endswith("_L") else "R" for pid in parent_records
        ],
        'True Grade': all_labels,
        'Predicted Grade': ensemble_preds,
        'Confidence (%)': np.round(ensemble_confs * 100, 2)
    })
    results_df['Correct'] = results_df['True Grade'] == results_df['Predicted Grade']
    
    csv_path = results_dir / f"predictions.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Detailed results saved to {csv_path}")

    cm_title_overall = f"Ensemble Confusion Matrix (Acc: {test_acc:.2%}, Bal Acc: {test_bal_acc:.2%})"
    fig_overall = create_confusion_matrix_figure(all_labels, ensemble_preds, num_classes=5, title=cm_title_overall)
    fig_overall.savefig(results_dir / f"cm_OVERALL", dpi=300)
    plt.close(fig_overall)

    #! Saving each images of the overall confusion matrix 
    img_save_dir = Path(f"./images_test/{timestamp}_fold{fold_handler.get_test_fold()}_{MODE}")
    if img_save_dir.exists():
        shutil.rmtree(img_save_dir)
    img_save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting {len(test_images_data)} visual overlays directly to '{img_save_dir}'...")

    for _, item in enumerate(tqdm(test_images_data, desc="Saving Overlays")):
        fig_img, ax_img = plt.subplots(figsize=(4, 4))
        ax_img.imshow(item["image"][:, :, 0], cmap="gray")

        title_color = "green" if item["true_label"] == item["pred_label"] else "red"
        ax_img.set_title(
            f'ID: {item["patient_id"]}\nTrue: {item["true_label"]} | Pred: {item["pred_label"]}\nEnsemble Conf: {item["confidence"]:.1%}', 
            color=title_color, 
            fontweight='bold', 
            fontsize=8
        )
        ax_img.axis('off')

        filename = img_save_dir / f"{item['patient_id']}_T{item['true_label']}_P{item['pred_label']}.png"
        fig_img.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(fig_img)

    logger.info("All visual test prediction overlays exported successfully!")
    logger.info("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing Classification")
    parser.add_argument("--timestamp", type=str, help="timestamp for the ensemble weights")
    parser.add_argument("--mode", type=str, default="BOTH", choices=["LEFT", "RIGHT", "BOTH"], help="Lateral standardization mode")
    args = parser.parse_args()

    test_classification(timestamp_override=args.timestamp, mode_override=args.mode)