import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet18, ResNet18_Weights, densenet121, DenseNet121_Weights

class RCNN(nn.Module):
    """
    Transfer learning object detection model using Faster R-CNN with a ResNet-50 backbone.
    """
    def __init__(self, num_classes: int=3):
        super(RCNN, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(weight="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.model.roi_heads.box_predictor.cls_score.in_features, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)

class Classification_ResNet(nn.Module):
    """
    Transfer learning image classification model using ResNet-18.
    """
    def __init__(self, num_classes: int=5, freeze_backbone: bool=False):
        super(Classification_ResNet, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            # for name, param in self.model.named_parameters():
            #     if "layer1" in name or "layer2" in name or "conv1" in name:
            #         param.requires_grad = False
            #     else:
            #         param.requires_grad = True
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class Classification_CNN(nn.Module):
    """
    A simple convolutional neural network (CNN) for image classification.
    """
    def __init__(self, num_classes: int=5, in_channels: int=1):
        super(Classification_CNN, self).__init__()
        # input size: (in_channels, 224, 224)
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#conv2d
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output: (32, 112, 112)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output: (64, 56, 56)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output: (128, 28, 28)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # output: (256, 14, 14)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512), 
            # output fc layer size is 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
            # output final class number
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Classification_DenseNet(nn.Module):
    def __init__(self, num_classes: int = 5, freeze_backbone: bool = False):
        super(Classification_DenseNet, self).__init__()
        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT) #, memory_efficient=True)

        selected_layer = ["conv0", "norm0"] + [f"denseblock{i}" for i in range(1, 4)] + [f"transition{i}" for i in range(1, 4)]
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                for layer in selected_layer:
                    if layer in name:
                        param.requires_grad = False
                        break

        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)