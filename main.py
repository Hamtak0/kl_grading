import argparse
import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import torch
import torchvision
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, balanced_accuracy_score

from dataset_handler import Fold_Handler, KLGradingDataset
from models import RCNN, Classification_DenseNet
from utils.config import load_toml_config
from utils.dicom_cut import read_dicom_image
from utils.logger import setup_logger
from utils.resize import extract_uniform_crop
from utils.transform import get_transform
from utils.visualize_rater import draw_box
from utils.metrics import create_confusion_matrix_figure

def main():
    parser = argparse.ArgumentParser(description="Knee Osteoarthritis full test")
    parser.add_argument("-c", "--config", default="configs/inference.toml", help="Path to inference settings config file")
    parser.add_argument("-t", "--target", type=str, choices=["KL", "OA", "kl", "oa"], help="Override Target (KL or OA)")
    parser.add_argument("-m", "--mode", type=str, choices=["LEFT", "RIGHT", "BOTH", "left", "right", "both"], help="Override Lateral Mode")
    parser.add_argument("-l", "--loss", type=str, choices=["CE", "MSE"], help="Override Loss Function")
    parser.add_argument("-rt", "--rcnn_ts", type=str, help="Override Detection Timestamp")
    parser.add_argument("-ct", "--class_ts", type=str, help="Override Classification Timestamp")
    parser.add_argument("-sc", "--score", type=float, help="Override Score Threshold")
    parser.add_argument("-iou", "--iou", type=float, help="Override IoU Threshold")
    parser.add_argument("-s", "--strategy", type=str, choices=["single_holdout", "kfold_blind", "kfold_nested"], help="Cross-validation architecture strategy")

    args = parser.parse_args()

    logger = setup_logger(name="Knee Osteoarthritis", log_file="inference.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cfg = load_toml_config(args.config)
    target = (args.target if args.target else cfg["inference"]["target"]).upper()
    mode = (args.mode if args.mode else cfg["inference"]["lateral_mode"]).upper()
    loss_fn = args.loss if args.loss else cfg["inference"]["loss_function"]
    rcnn_ts = args.rcnn_ts if args.rcnn_ts else cfg["inference"]["rcnn_timestamp"]
    cls_ts = args.class_ts if args.class_ts else cfg["inference"]["class_timestamp"]
    strategy = args.strategy if args.strategy else cfg["inference"]["strategy"]

    score_threshold = args.score if args.score is not None else cfg["inference"]["score_threshold"]
    iou_threshold = args.iou if args.iou is not None else cfg["inference"]["iou_threshold"]

    logger.info(f"Active settings -> Target: {target}, Mode: {mode}, Strategy: {strategy}, Timestamp: RCNN {rcnn_ts} & CLS {cls_ts}, Loss function: {loss_fn}")

    env_cfg = load_toml_config(Path("configs/config.toml"))
    try:
        dataset = KLGradingDataset(
            root=env_cfg["dataset"]["root_dir"],
            include_grades=True,
            grade_path=env_cfg["dataset"]["excel_file"],
        )
    except Exception as e:
        logger.error(f"Failed to mount KLGradingDataset: {e}")
        sys.exit(1)

    fold_handler = Fold_Handler(dataset)
    all_folds = fold_handler.get_all_folds()

    cls_transform = get_transform(rotation=0, jitter=0, sharpness=0, is_train=False).to(device)

    output_dir = Path(f"./outputs/{rcnn_ts}_{cls_ts}_{target}_{mode}_{strategy}_{loss_fn}/")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_true = []
    all_pred = []
    all_confs = []
    all_pids = []

    loop_folds = [fold_handler.get_test_fold()] if strategy == "single_holdout" else all_folds

    for test_fold in loop_folds:
        logger.info(f"{'-' * 20} Evaluating test fold: {test_fold} {'-' * 20}")

        det_model = RCNN(num_classes=3).to(device)
        det_weight = Path(f"./weights/knee_rcnn_{rcnn_ts}_fold{test_fold}.pth")
        try:
            det_model.load_state_dict(torch.load(det_weight, map_location=device, weights_only=True))
            det_model.eval()
        except Exception as e:
            logger.error(f"Failed to load detection model weights for fold {test_fold}: {e}")
            continue

        if target == "OA":
            num_classes = 2
        elif target == "KL":
            if loss_fn == "MSE": num_classes = 1
            elif loss_fn == "CE": num_classes = 5

        cls_models = []
        if strategy == "single_holdout":
            logger.info("Loading for single holdout strategy")
            for fold in fold_handler.get_cv_folds():
                model = Classification_DenseNet(num_classes=num_classes).to(device)
                weight_path = Path(f"./weights/knee_class_{cls_ts}_{target}_fold{fold}_{mode}.pth")
                if weight_path.exists():
                    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
                    model.eval()
                    cls_models.append(model)
        else:
            model = Classification_DenseNet(num_classes=num_classes).to(device)
            weight_path = Path(f"./weights/knee_class_{cls_ts}_{target}_fold{test_fold}_{mode}.pth")
            if weight_path.exists():
                model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
                model.eval()
                cls_models.append(model)

        if not cls_models:
            logger.error(f"No valid model weights found for fold {test_fold}. Skipping.")
            continue

        test_idx = [i for i, pid in enumerate(dataset.patient_ids) if fold_handler.get_fold(pid) == test_fold]

        fold_true = []
        fold_pred = []
        fold_confs = []
        fold_pids = []

        with torch.no_grad():
            for idx in tqdm(test_idx, desc=f"Testing fold {test_fold}"):
                patient_id = dataset.patient_ids[idx]
                dicom_file = dataset.dicom[idx]
                dicom_path = (
                    Path(env_cfg["dataset"]["root_dir"])
                    / "bilateral_standing_AP"
                    / dicom_file
                )

                patient_grades = dataset.labels_dict.get(patient_id, {"L": [-1, -1], "R": [-1, -1]})

                ds, image_array = read_dicom_image(dicom_path)
                img_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1).to(device) # change to (3, H, W)

                det_preds = det_model([img_tensor])[0]
                boxes = det_preds["boxes"]
                scores = det_preds["scores"]
                labels = det_preds["labels"]

                mask = scores > score_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]

                keep_idx = torchvision.ops.nms(boxes, scores, iou_threshold)
                boxes = boxes[keep_idx]
                scores = scores[keep_idx]
                labels = labels[keep_idx]

                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(image_array, cmap="gray")
                ax.set_title(f"Patient {patient_id} (test fold {test_fold})", fontsize=14, pad=20)
                ax.axis("off")

                for box, score, label in zip(boxes, scores, labels):
                    if label.item() == 1:  # is left
                        side_str = "L"
                        box_color = "red"
                    elif label.item() == 2:  # is right
                        side_str = "R"
                        box_color = "blue"

                    true_kl, true_oa = patient_grades.get(side_str, [-1, -1])
                    if target == "KL":
                        true_label = true_kl
                    elif target == "OA":
                        true_label = true_oa

                    crop = extract_uniform_crop(img_tensor, box.cpu().numpy())

                    if mode == "LEFT" and label.item() == 2:
                        crop = torch.flip(crop, dims=[2])
                    elif mode == "RIGHT" and label.item() == 1:
                        crop = torch.flip(crop, dims=[2])

                    crop = cls_transform(crop).unsqueeze(0)

                    ensemble_outputs = []
                    for cls_model in cls_models:
                        with torch.amp.autocast('cuda'):
                            out = cls_model(crop)
                        out = out.float()

                        if target == "KL" and loss_fn == "MSE":
                            ensemble_outputs.append(out.squeeze(1))
                        elif target == "KL" or loss_fn == "CE":
                            ensemble_outputs.append(torch.nn.functional.softmax(out, dim=1))

                    avg_output = torch.stack(ensemble_outputs).mean(dim=0)

                    if target == "KL":
                        if loss_fn == "MSE":
                            pred_class = torch.round(avg_output).clamp(0, 4).long().item()
                            distance = abs(avg_output.item() - pred_class)
                            confidence = max(0.0, min(1.0, 1.0 - distance))
                        elif loss_fn == "CE":
                            confidence, pred_idx = torch.max(avg_output, 1)
                            confidence = confidence.item()
                            pred_class = pred_idx.item()

                        pred_text = f"Grade {pred_class}"
                        true_text = f"Grade {true_label}" if true_label != -1 else "Unknown"
                    
                    elif target == "OA":
                        confidence, pred_idx = torch.max(avg_output, 1)
                        confidence = confidence.item()
                        pred_class = pred_idx.item()

                        pred_text = "OA" if pred_class == 1 else "Healthy"
                        true_text = "OA" if true_label == 1 else ("Healthy" if true_label == 0 else "Unknown")
                    
                    if true_label != -1:
                        fold_true.append(true_label)
                        fold_pred.append(pred_class)
                        fold_confs.append(confidence)
                        fold_pids.append(f"{patient_id}_{side_str}")
                    
                    if true_text == pred_text:
                        box_color = "green"

                    title = f"{side_str} ({score:.2f})\n{target} True: {true_text} Pred: {pred_text} ({confidence:.2%})"
                    draw_box(ax, color=box_color, title=title, box=box.cpu().numpy())

                out_path = output_dir / f"{patient_id}_fold{test_fold}.png"
                fig.savefig(out_path, bbox_inches="tight", dpi=300)
                plt.close(fig)

        if len(fold_true) > 0:
            f_acc = accuracy_score(fold_true, fold_pred)
            f_bal_acc = balanced_accuracy_score(fold_true, fold_pred)
            f1 = f1_score(fold_true, fold_pred, average="weighted")

            logger.info(f"-- Model {test_fold} Metrics --")
            logger.info(f"Accuracy: {f_acc:.2%}")
            logger.info(f"Balanced Accuracy: {f_bal_acc:.2%}")
            logger.info(f"Weighted F1 Score: {f1:.4f}")

            all_true.extend(fold_true)
            all_pred.extend(fold_pred)
            all_confs.extend(fold_confs)
            all_pids.extend(fold_pids)
        else:
            logger.warning(f"No valid predictions for fold {test_fold}. Skipping metrics.")

    logger.info("-" * 20 + f" Metrics Report ({strategy}) " + "-" * 20)

    if len(all_true) > 0:
        acc = accuracy_score(all_true, all_pred)
        bal_acc = balanced_accuracy_score(all_true, all_pred)
        f1 = f1_score(all_true, all_pred, average="weighted")

        logger.info(f"Total knee evaluated: {len(all_true)}")
        logger.info(f"Overall Accuracy: {acc:.2%}")
        logger.info(f"Balanced Accuracy: {bal_acc:.2%}")
        logger.info(f"Weighted F1 Score: {f1:.4f}")

        target_names = [f"Grade {i}" for i in range(5)] if target == "KL" else ["Healthy (0)", "OA (1)"]
        report = classification_report(all_true, all_pred, target_names=target_names, zero_division=0)
        logger.info(f"\n{report}")

        cm_title = f"Full pipeline {target} ({strategy})\nAcc: {acc:.2%}, Bal Acc: {bal_acc:.2%}, F1: {f1:.4f}"
        fig_cm = create_confusion_matrix_figure(all_true, all_pred, target=target, title=cm_title)
        cm_path = output_dir / f"confusion_matrix_{target}.png"
        fig_cm.savefig(cm_path, bbox_inches="tight", dpi=300)
        plt.close(fig_cm)

        results_df = pd.DataFrame({
            "Patient_ID": all_pids,
            "Side": ["L" if pid.endswith("_L") else "R" for pid in all_pids],
            f"True {target}": all_true,
            f"Predicted {target}": all_pred,
            "Confidence (%)": np.round(np.array(all_confs) * 100, 2)
        })
        results_df['Correct'] = results_df[f'True {target}'] == results_df[f'Predicted {target}']
        
        csv_path = output_dir / f"table_{target}.csv"
        results_df.to_csv(csv_path, index=False)
        
        logger.info(f"Metrics complete! Matrix and CSV saved to {output_dir}")
    else:
        logger.warning("No ground truth labels were found. Skipping metrics.")

if __name__ == "__main__":
    main()
