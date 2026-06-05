from core.classification.models import RCNN, Classification_ResNet, Classification_DenseNet

# model = Classification_ResNet(num_classes=5, freeze_backbone=True)
# model = Classification_DenseNet(num_classes=5, freeze_backbone=True)
model = RCNN(num_classes=3)

# Iterate over named parameters
for name, param in model.named_parameters():
    print(f"Name: {name}\tShape: {param.shape}\tRequires Grad: {param.requires_grad}")