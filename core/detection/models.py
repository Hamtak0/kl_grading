import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class RCNN(nn.Module):
    """
    Transfer learning object detection model using Faster R-CNN with a ResNet-50 backbone.
    """
    def __init__(self, num_classes: int=3):
        super(RCNN, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.model.roi_heads.box_predictor.cls_score.in_features, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)