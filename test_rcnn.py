import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from pathlib import Path

from utils.logger import setup_logger
from utils.visualize_rater import draw_box
from utils.transform import get_transform
from dataset_handler import KLGradingDataset
from models import RCNN

def test():
    logger = setup_logger(name="Knee Project", log_file="rcnn_testing.log")
    logger.info("Starting inference on the test set...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running on device: {device}")

    num_classes = 3
    model = RCNN(num_classes=num_classes)
    knee_weight_name = "knee_rcnn_20260424_121014.pth"
    weight_path = Path("./weights/") / knee_weight_name

    try:
        # weights_only=True is a security best practice in modern PyTorch
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        logger.info(f"Successfully loaded weights from {weight_path}")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        return
    model.to(device)

    model.eval()
    root_dir = "./dataset"
    try:
        full_dataset = KLGradingDataset(root=root_dir, transforms=get_transform())
        test_indices = torch.load("./weights/test_indices.pth", weights_only=True)
        test_subset = Subset(full_dataset, test_indices)

        logger.info(f"Evaluating on {len(test_subset)} unseen images.\n")
    except Exception as e:
        logger.error(f"Failed to load test data. Did you run train.py first? Error: {e}")
        return

    num_to_visualize = min(5, len(test_subset))

    for i in range(num_to_visualize):
        img_tensor, target = test_subset[i]

        # --- INFERENCE ---
        with torch.no_grad():
            prediction = model([img_tensor.to(device)])[0]

        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        # --- FILTERING ---
        best_left = {"box": None, "score": 0.0}
        best_right = {"box": None, "score": 0.0}
        confidence_threshold = 0.50

        for box, label, score in zip(boxes, labels, scores):
            if label == 1 and score > confidence_threshold and score > best_left["score"]:
                best_left = {"box": box, "score": score}
            elif label == 2 and score > confidence_threshold and score > best_right["score"]:
                best_right = {"box": box, "score": score}

        # --- PLOTTING ---
        # The image tensor is [3, H, W]. Slice the first channel for grayscale display.
        img_display = img_tensor[0].cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img_display, cmap="gray")
        ax.set_title(f"Test Image {i+1} / {num_to_visualize}")

        # Ground Truth (Green / Cyan)
        gt_boxes = target['boxes'].numpy()
        gt_labels = target['labels'].numpy()
        for box, label in zip(gt_boxes, gt_labels):
            color = "green" if label == 1 else "cyan"
            # Using your imported draw_box function!
            draw_box(ax, color=color, title=f"GT {'Left' if label==1 else 'Right'}", box=box)

        # Predictions (Red / Blue)
        if best_left["box"] is not None:
            draw_box(ax, color="red", title=f"Pred Left ({best_left['score']:.2f})", box=best_left["box"])
        if best_right["box"] is not None:
            draw_box(ax, color="blue", title=f"Pred Right ({best_right['score']:.2f})", box=best_right["box"])

        plt.legend(loc="upper right")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    test()
