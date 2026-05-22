import argparse
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset_handler import KLGradingDataset, Fold_Handler
from models import RCNN
from utils.logger import setup_logger
from utils.config import load_toml_config
from utils.seed_setup import set_seed
from utils.transform import collate_fn
from utils.visualize_rater import draw_box

def test_rcnn(timestamp_override: str = None):
    set_seed(42)
    logger = setup_logger(name="Knee Project", log_file="rcnn_testing.log")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = timestamp_override if timestamp_override else "20260519_112204"
    logger.info(f"Testing object detection (Timestamp: {timestamp})")
    logger.info(f"Running on device: {device}")

    env_cfg = load_toml_config("configs/config.toml")
    try:
        full_dataset = KLGradingDataset(
            root=env_cfg["dataset"]["root_dir"],
            include_grades=True,
            grade_path=env_cfg["dataset"]["excel_file"]
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    fold_handler = Fold_Handler(full_dataset)
    test_idx = []

    for i in range(len(full_dataset)):
        patient_id = full_dataset.patient_ids[i]
        if fold_handler.get_fold(patient_id) == fold_handler.get_test_fold():
            test_idx.append(i)

    test_subset = Subset(full_dataset, test_idx)
    data_loader_test = DataLoader(
        test_subset,
        batch_size=1,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )

    models = []
    num_classes = 3
    for fold in fold_handler.get_cv_folds():
        model = RCNN(num_classes=num_classes).to(device)
        weight_path = Path(f"./weights/knee_rcnn_{timestamp}_fold{fold}.pth")
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            model.eval()
            models.append(model)
        except Exception as e:
            logger.error(f"Could not load R-CNN weights at {weight_path}. Error: {e}")
            return

    if not models:
        logger.error("No models loaded. Aborting...")
        return

    output_dir = Path(f"./cropped_test/{timestamp}_fold0")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running inference on {len(test_subset)} unseen X-rays")

    # Non-Maximum Suppression (NMS)
    IOU_THRESHOLD = 0.5
    SCORE_THRESHOLD = 0.7

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader_test)):
            images = list(image.to(device) for image in images)

            # since batch_size is 1
            image_tensor = images[0]

            all_boxes = []
            all_scores = []
            all_labels = []

            for model in models:
                prediction = model([image_tensor])[0]
                all_boxes.append(prediction['boxes'])
                all_scores.append(prediction['scores'])
                all_labels.append(prediction['labels'])

            cat_boxes = torch.cat(all_boxes, dim=0)
            cat_scores = torch.cat(all_scores, dim=0)
            cat_labels = torch.cat(all_labels, dim=0)

            # filtering the low confident 
            mask = cat_scores > SCORE_THRESHOLD
            cat_boxes = cat_boxes[mask]
            cat_scores = cat_scores[mask]
            cat_labels = cat_labels[mask]

            # apply nms to fuse overlapping boxes
            keep_idx = torchvision.ops.nms(cat_boxes, cat_scores, IOU_THRESHOLD)
            final_boxes = cat_boxes[keep_idx].cpu().numpy()
            final_scores = cat_scores[keep_idx].cpu().numpy()
            final_labels = cat_labels[keep_idx].cpu().numpy()

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(image_tensor[0].cpu().numpy(), cmap="gray")
            
            patient_id = full_dataset.bounding[test_idx[batch_idx]].split(".")[0]
            ax.set_title(f"Ensemble R-CNN Prediction: {patient_id}")
            ax.axis('off')

            for box, score, label in zip(final_boxes, final_scores, final_labels):
                color = "red" if label == 1 else "blue" # 1 = Left, 2 = Right
                side = "L" if label == 1 else "R"
                draw_box(ax, color=color, title=f"{side} ({score:.2f})", box=box)

            fig.savefig(output_dir / f"{patient_id}_rcnn_ensemble.png", bbox_inches='tight', dpi=150)
            plt.close(fig)

    logger.info(f"Testing complete! Visualizations saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test knee detection")
    parser.add_argument("--timestamp", type=str, help="Timestamp of the trained object detection model")
    args = parser.parse_args()
    test_rcnn(timestamp_override=args.timestamp)
