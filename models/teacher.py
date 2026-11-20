import torch
import torch.nn as nn
from torchvision.models import resnet50


class ResNet50Teacher(nn.Module):
    """CIFAR-100 adapted ResNet-50 exposing 4 intermediate stage feature maps."""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        backbone = resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = backbone.layer1  # 256 x 32 x 32
        self.layer2 = backbone.layer2  # 512 x 16 x 16
        self.layer3 = backbone.layer3  # 1024 x 8 x 8
        self.layer4 = backbone.layer4  # 2048 x 4 x 4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(2048, num_classes)

        self.feature_channels = [256, 512, 1024, 2048]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        out = torch.flatten(self.avgpool(f4), 1)
        logits = self.fc(out)
        if return_features:
            return logits, [f1, f2, f3, f4]
        return logits
