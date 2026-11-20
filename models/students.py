import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class MobileNetV2Student(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.stage1 = InvertedResidual(32, 16, stride=1, expand_ratio=1)
        self.stage2 = nn.Sequential(
            InvertedResidual(16, 24, stride=2, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6),
        )
        self.stage3 = nn.Sequential(
            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
        )
        self.stage4 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 160, stride=1, expand_ratio=6),
            nn.Conv2d(160, 320, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU6(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(320, num_classes))
        self.feature_channels = [16, 24, 32, 320]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        logits = self.classifier(torch.flatten(self.pool(f4), 1))
        if return_features:
            return logits, [f1, f2, f3, f4]
        return logits


class FireModule(nn.Module):
    def __init__(self, in_ch: int, squeeze_ch: int, expand1x1_ch: int, expand3x3_ch: int):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_ch, squeeze_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(squeeze_ch),
            nn.ReLU(inplace=True),
        )
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_ch, expand1x1_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand1x1_ch),
            nn.ReLU(inplace=True),
        )
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_ch, expand3x3_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(expand3x3_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.squeeze(x)
        return torch.cat([self.expand1x1(s), self.expand3x3(s)], dim=1)


class SqueezeNetStudent(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            FireModule(64, 16, 64, 64),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
        )
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
        )
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feature_channels = [128, 256, 384, 512]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        logits = torch.flatten(self.classifier(f4), 1)
        if return_features:
            return logits, [f1, f2, f3, f4]
        return logits


class MBConvLite(nn.Module):
    def __init__(self, inp: int, oup: int, kernel_size: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden_dim = inp * expand_ratio
        self.use_res = stride == 1 and inp == oup
        padding = kernel_size // 2

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride, padding=padding, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x) if self.use_res else self.block(x)


class EfficientNetLite0Student(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.stage1 = MBConvLite(32, 16, kernel_size=3, stride=1, expand_ratio=1)
        self.stage2 = nn.Sequential(
            MBConvLite(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            MBConvLite(24, 24, kernel_size=3, stride=1, expand_ratio=6),
        )
        self.stage3 = nn.Sequential(
            MBConvLite(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            MBConvLite(40, 40, kernel_size=5, stride=1, expand_ratio=6),
        )
        self.stage4 = nn.Sequential(
            MBConvLite(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            MBConvLite(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConvLite(112, 192, kernel_size=5, stride=1, expand_ratio=6),
            nn.Conv2d(192, 320, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU6(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(320, num_classes))
        self.feature_channels = [16, 24, 40, 320]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        logits = self.classifier(torch.flatten(self.pool(f4), 1))
        if return_features:
            return logits, [f1, f2, f3, f4]
        return logits


def get_student_model(name: str, num_classes: int = 100) -> nn.Module:
    registry = {
        "mobilenetv2": MobileNetV2Student,
        "squeezenet": SqueezeNetStudent,
        "efficientnet_lite0": EfficientNetLite0Student,
    }
    key = name.lower()
    if key not in registry:
        raise ValueError(f"Unknown student '{name}'. Choose from {list(registry.keys())}.")
    return registry[key](num_classes=num_classes)
