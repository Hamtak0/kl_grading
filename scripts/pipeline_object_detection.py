import argparse
from pathlib import Path
import sys

from core.object_detection.train import train_rcnn
from core.object_detection.test import test_rcnn
from utils.config import load_toml_config

def pipeline_object_detection():
    parser = argparse.ArgumentParser(description="Full object detection pipeline model training and testing")
    parser.add_argument("--config", type=str, default="configs/config.toml", help="Path to baseline TOML configuration")
    parser.add_argument("-e", "--epochs", type=int, help="Override number of epochs for training")
    parser.add_argument("-b", "--batch", type=int, help="Override batch size for training")
    parser.add_argument("-sc", "--score", type=float, help="Override Score Threshold")
    parser.add_argument("-iou", "--iou", type=float, help="Override IoU Threshold")
    parser.add_argument("-s", "--strategy", type=str, choices=["single_holdout", "kfold_blind", "kfold_nested"], help="Override Cross-validation architecture strategy")
    args = parser.parse_args()

    cfg = load_toml_config(Path(args.config))

    base_epochs = args.epochs if args.epochs else cfg["detection_defaults"]["num_epochs"]
    base_batch_size = args.batch if args.batch else cfg["detection_defaults"]["batch_size"]
    score_threshold = args.score if args.score else cfg["detection_defaults"]["score_threshold"]
    iou_threshold = args.iou if args.iou else cfg["detection_defaults"]["iou_threshold"]
    strategy = args.strategy if args.strategy else cfg["experiment"]["strategy"]

    print(f"Target Configuration: {args.config}")
    print(f"Strategy: {strategy} | Epochs: {base_epochs} | Score Threshold: {score_threshold} | IoU Threshold: {iou_threshold}")

    print(f"-- Phase 1: Object Detection Model Training")
    timestamp = train_rcnn(epochs_override=base_epochs, batch_size_override=base_batch_size, strategy=strategy)

    if not timestamp:
        print("Training failed to yield a valid model timestamp. Aborting.")
        sys.exit(1)

    print(f"-- Phase 2: Evaluation with test fold ({strategy})")
    test_rcnn(score_threshold=score_threshold, iou_threshold=iou_threshold, strategy=strategy, timestamp_override=timestamp)

    print(f"Pipeline successfully completed for {strategy}!")

if __name__ == "__main__":
    pipeline_object_detection()