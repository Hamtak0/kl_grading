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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score
from tqdm import tqdm

from utils.seed_setup import set_seed
from utils.logger import setup_logger
from utils.metrics import create_confusion_matrix_figure
from utils.transform import get_transform, unnormalize_tensor
from utils.config import load_toml_config
from dataset_handler import CachedKneeDataset, TransformWrapper, Fold_Handler
from models import Classification_DenseNet

def test_classification(timestamp_override: str = None, mode_override: str = "BOTH", target: str = "OA", loss_fn: str = "CE", strategy: str = "kfold_nested"):
    set_seed(42)

    logger = setup_logger(name="KneeTest", log_file="classifier_testing.log")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Starting Testing on device: {device}")

    timestamp = timestamp_override if timestamp_override else "20260511_154935"
    MODE = mode_override.upper()
    logger.info(f"Test for {target} ({loss_fn}), strategy: {strategy}, timestamp: {timestamp}, mode: {MODE}")

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
    all_folds = fold_handler.get_all_folds()

    all_labels = []
    all_preds = []
    all_confs = []
    all_pids = []
    test_images_data = []
    gpu_test_transform = get_transform(rotation=0, jitter=0, sharpness=0, is_train=False).to(device)

    if target == "KL":
        if loss_fn == "MSE": num_classes = 1
        elif loss_fn == "CE": num_classes = 5
    elif target == "OA": num_classes = 2
    target_names = [f"Grade {i}" for i in range(5)] if target == "KL" else ["Healthy (0)", "OA (1)"]

    if strategy == "single_holdout":
        test_fold = fold_handler.get_test_fold()
        logger.info(f"Executing ensemble with test fold {test_fold}.")

        models = []
        cv_folds = fold_handler.get_cv_folds()
        for fold in cv_folds:
            model = Classification_DenseNet(num_classes=num_classes).to(device)
            weight_path = Path(f"./weights/knee_class_{timestamp}_{target}_fold{fold}_{MODE}.pth")
            if weight_path.exists():
                model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
                model.eval()
                models.append(model)
            else:
                logger.warning(f"Model weights not found for fold {fold} at {weight_path}. Skipping this fold.")

        if not models:
            logger.error("No valid models found for testing. Aborting.")
            return

        test_idx = [i for i, sample in enumerate(cv_classifier_dataset.samples) if fold_handler.get_fold(sample["patient_id"].split("_")[0]) == test_fold]
        test_subset = Subset(cv_classifier_dataset, test_idx)
        test_dataset = TransformWrapper(test_subset, mode=MODE)
        data_loader_test = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=3, pin_memory=True)

        with torch.no_grad():
            for images, kl, oa, pids in tqdm(data_loader_test, desc=f"Ensemble testing with fold-{test_fold}"):
                if target == "KL": labels = kl.to(device)
                elif target == "OA": labels = oa.to(device)
                images = gpu_test_transform(images.to(device))

                ensemble_outputs = []
                for model in models:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                    outputs = outputs.float()

                    if target == "KL" and loss_fn == "MSE":
                        ensemble_outputs.append(outputs.squeeze(1))
                    elif target == "OA" or loss_fn == "CE":
                        probabilities = F.softmax(outputs, dim=1)
                        ensemble_outputs.append(probabilities)

                avg_outputs = torch.mean(torch.stack(ensemble_outputs), dim=0)

                if target == "KL" and loss_fn == "MSE":
                    predicted = torch.round(avg_outputs).clamp(0, 4).long()
                    distances = torch.abs(avg_outputs - predicted.float()) 
                    confidences = torch.clamp(1.0 - distances, min=0.0, max=1.0)
                elif target == "OA" or loss_fn == "CE":
                    probabilities = F.softmax(avg_outputs, dim=1)
                    confidences, predicted = torch.max(probabilities, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_confs.extend(confidences.cpu().numpy())
                all_pids.extend(pids)

                images_np = unnormalize_tensor(images.cpu())
                for i in range(len(labels)):
                    test_images_data.append({
                        "image": images_np[i],
                        "true_label": labels[i].item(),
                        "pred_label": predicted[i].item(),
                        "confidence": confidences[i].item(),
                        "patient_id": pids[i]
                    })

    # Evaluate K-fold blind and nested strategies
    else:
        for test_fold in all_folds:
            logger.info(f"Testing with fold {test_fold}")

            test_idx = [i for i, sample in enumerate(cv_classifier_dataset.samples) if fold_handler.get_fold(sample["patient_id"].split("_")[0]) == test_fold]
            test_subset = Subset(cv_classifier_dataset, test_idx)
            test_dataset = TransformWrapper(test_subset, mode=MODE)
            data_loader_test = DataLoader(
                test_dataset,
                batch_size=2,
                shuffle=False,
                num_workers=3,
                pin_memory=True
            )

            model = Classification_DenseNet(num_classes=num_classes).to(device)
            weight_path = Path(f"./weights/knee_class_{timestamp}_{target}_fold{test_fold}_{MODE}.pth")

            if not weight_path.exists():
                logger.warning(f"File {weight_path} missing. Skipping fold {test_fold}.")
                continue

            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            model.eval()
    
            with torch.no_grad():
                for images, kl, oa, pids in tqdm(data_loader_test, desc=f"Testing with fold-{test_fold}"):
                    if target == "KL": labels = kl.to(device)
                    elif target == "OA": labels = oa.to(device)
                    images = gpu_test_transform(images.to(device))

                    with torch.amp.autocast('cuda'):
                        outputs = model(images)

                    outputs = outputs.float()

                    if target == "KL" and loss_fn == "MSE":
                        outputs_mse = outputs.squeeze(1)
                        predicted = torch.round(outputs_mse).clamp(0, 4).long()
                        distances = torch.abs(outputs_mse - predicted.float())
                        confidences = torch.clamp(1.0 - distances, min=0.0, max=1.0)
                    elif target == "OA" or loss_fn == "CE":
                        probabilities = F.softmax(outputs, dim=1)
                        confidences, predicted = torch.max(probabilities, 1)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                    all_confs.extend(confidences.cpu().numpy())
                    all_pids.extend(pids)

                    images_np = unnormalize_tensor(images.cpu())
                    for i in range(len(labels)):
                        test_images_data.append({
                            "image": images_np[i],
                            "true_label": labels[i].item(),
                            "pred_label": predicted[i].item(),
                            "confidence": confidences[i].item(),
                            "patient_id": pids[i]
                        })

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_confs = np.array(all_confs)
    all_pids = np.array(all_pids)

    if len(all_labels) == 0:
        logger.error("No test predictions were collected. Aborting evaluation.")
        return

    test_acc = accuracy_score(all_labels, all_preds)
    test_bal_acc = balanced_accuracy_score(all_labels, all_preds)

    logger.info(f"-- Final test accuracy ({target}): {test_acc:.2%}")
    logger.info(f"-- Final balanced accuracy ({target}): {test_bal_acc:.2%}")

    logger.info(f"Classification {target} report {strategy}:")
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    logger.info("\n" + report)

    # creates results directory for grouping test together
    os.makedirs("./results", exist_ok=True)
    results_dir = Path(f"./results/{timestamp}_{target}_{MODE}_{strategy}")
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Consolidating all performance into: {results_dir}")

    # Bilateral subgroup export
    logger.info(f"Bilateral subgroup analysis (Standardization: {MODE}):")
    left_mask = np.array([pid.endswith("_L") for pid in all_pids])
    right_mask = np.array([pid.endswith("_R") for pid in all_pids])

    if np.any(left_mask):
        l_labels = all_labels[left_mask]
        l_preds = all_preds[left_mask]
        l_acc = accuracy_score(l_labels, l_preds)
        l_bal_acc = balanced_accuracy_score(l_labels, l_preds)
        logger.info(f"Left knees (count: {np.sum(left_mask)}) - Accuracy: {l_acc:.2%}, Balanced Accuracy: {l_bal_acc:.2%}")

        cm_title_l = f"Left knees ({MODE})\nAccuracy: {l_acc:.2f}, Balance Accurary: {l_bal_acc:.2f}"
        fig_l = create_confusion_matrix_figure(l_labels, l_preds, target=target, title=cm_title_l)
        fig_l.savefig(results_dir / f"cm_LEFT.png", dpi=300)
        plt.close(fig_l)
    else:
        logger.warning("No left knees found in the test set")

    if np.any(right_mask):
        r_labels = all_labels[right_mask]
        r_preds = all_preds[right_mask]
        r_acc = accuracy_score(r_labels, r_preds)
        r_bal_acc = balanced_accuracy_score(r_labels, r_preds)
        logger.info(f"Right knees (count: {np.sum(right_mask)}) - Accuracy: {r_acc:.2%}, Balanced Accuracy: {r_bal_acc:.2%}")

        cm_title_r = f"Left knees ({MODE})\nAccuracy: {r_acc:.2f}, Balance Accurary: {r_bal_acc:.2f}"
        fig_r = create_confusion_matrix_figure(r_labels, r_preds, target=target, title=cm_title_r)
        fig_r.savefig(results_dir / f"cm_RIGHT.png", dpi=300)
        plt.close(fig_r)
    else:
        logger.warning("No right knees found in the test set")

    results_df = pd.DataFrame({
        'Patient Knee': all_pids,
        'Side': ["L" if pid.endswith("_L") else "R" for pid in all_pids],
        f'True {target}': all_labels,
        f'Predicted {target}': all_preds,
        'Confidence (%)': np.round(all_confs * 100, 2)
    })
    results_df['Correct'] = results_df[f'True {target}'] == results_df[f'Predicted {target}']
    
    # CSV export
    csv_path = results_dir / f"predictions_{target}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Detailed results saved to {csv_path}")

    cm_title_overall = f"Confusion Matrix {target} (Acc: {test_acc:.2%}, Bal Acc: {test_bal_acc:.2%})"
    fig_overall = create_confusion_matrix_figure(all_labels, all_preds, target=target, title=cm_title_overall)
    fig_overall.savefig(results_dir / f"cm_OVERALL", dpi=300)
    plt.close(fig_overall)

    # Saving each images of the overall confusion matrix 
    img_save_dir = Path(f"./images_test/{timestamp}_{target}_{MODE}_{strategy}")
    if img_save_dir.exists():
        shutil.rmtree(img_save_dir)
    img_save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting {len(test_images_data)} visual overlays directly to '{img_save_dir}'...")

    for _, item in enumerate(tqdm(test_images_data, desc="Saving Overlays")):
        fig_img, ax_img = plt.subplots(figsize=(4, 4))
        ax_img.imshow(item["image"][:, :, 0], cmap="gray")

        title_color = "green" if item["true_label"] == item["pred_label"] else "red"
        ax_img.set_title(
            f'ID: {item["patient_id"]}\n{target} True: {item["true_label"]} | Pred: {item["pred_label"]}\nConf: {item["confidence"]:.1%}', 
            color=title_color, 
            fontweight='bold', 
            fontsize=8
        )
        ax_img.axis('off')

        filename = img_save_dir / f"{item['patient_id']}_T{item['true_label']}_P{item['pred_label']}.png"
        fig_img.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(fig_img)

    logger.info("All visual test prediction overlays exported successfully!")
    logger.info("=" * 20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing Classification")
    parser.add_argument("-ts", "--timestamp", type=str, help="timestamp for the ensemble weights")
    parser.add_argument("-m", "--mode", type=str, choices=["LEFT", "RIGHT", "BOTH"], help="Lateral standardization mode")
    parser.add_argument("-t", "--target", type=str, choices=["KL", "OA"], help="Target of the model (KL grading or Osteoarthritis separation)")
    parser.add_argument("-l", "--loss", type=str, choices=["CE", "MSE"], help="Loss function to be trained with")
    parser.add_argument("-s", "--strategy", type=str, choices=["single_holdout", "kfold_blind", "kfold_nested"], help="Cross-validation architecture to use")
    args = parser.parse_args()
    test_classification(timestamp_override=args.timestamp, mode_override=args.mode, target=args.target, loss_fn=args.loss, strategy=args.strategy)