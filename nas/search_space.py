from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn


SEARCH_SPACE = {
    "num_blocks": [3, 4, 5, 6, 7],
    "channels": [16, 24, 32, 48, 64, 96, 128],
    "kernel_sizes": [3, 5, 7],
    "block_types": ["standard_conv", "dw_sep_conv", "inverted_residual"],
    "activations": ["relu", "relu6", "hardswish", "gelu"],
    "input_resolutions": [24, 28, 32],
}


def get_activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU(inplace=True),
        "relu6": nn.ReLU6(inplace=True),
        "hardswish": nn.Hardswish(inplace=True),
        "gelu": nn.GELU(),
    }[name]


class SearchableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, block_type: str, act_name: str):
        super().__init__()
        padding = kernel_size // 2
        self.use_res = stride == 1 and in_ch == out_ch

        if block_type == "standard_conv":
            self.op = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_ch),
                get_activation(act_name),
            )
        elif block_type == "dw_sep_conv":
            self.op = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride, padding=padding, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                get_activation(act_name),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                get_activation(act_name),
            )
        elif block_type == "inverted_residual":
            hidden = in_ch * 4
            self.op = nn.Sequential(
                nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                get_activation(act_name),
                nn.Conv2d(hidden, hidden, kernel_size, stride=stride, padding=padding, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                get_activation(act_name),
                nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            raise ValueError(f"Unsupported block_type: {block_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.op(x)
        return x + out if self.use_res else out


class SearchableCNN(nn.Module):
    def __init__(self, config: Dict[str, Any], num_classes: int = 10):
        super().__init__()
        self.config = config
        self.input_resolution = int(config["input_resolution"])
        num_blocks = int(config["num_blocks"])

        stem_ch = int(config["channels"][0])
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            get_activation(config["activations"][0]),
        )

        downsample_indices = set(np.linspace(0, num_blocks - 1, min(3, num_blocks), dtype=int).tolist())

        blocks = []
        in_ch = stem_ch
        for i in range(num_blocks):
            out_ch = int(config["channels"][i])
            k = int(config["kernel_sizes"][i])
            b_type = config["block_types"][i]
            act = config["activations"][i]
            stride = 2 if i in downsample_indices else 1
            blocks.append(SearchableBlock(in_ch, out_ch, k, stride, b_type, act))
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.15), nn.Linear(in_ch, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = torch.flatten(self.pool(x), 1)
        return self.classifier(x)


class ArchitectureGenerator:
    @staticmethod
    def sample_config(trial) -> Dict[str, Any]:
        num_blocks = trial.suggest_int("num_blocks", 3, 7)
        input_resolution = trial.suggest_categorical("input_resolution", SEARCH_SPACE["input_resolutions"])

        channels, kernel_sizes, block_types, activations = [], [], [], []
        for i in range(num_blocks):
            channels.append(trial.suggest_categorical(f"ch_b{i}", SEARCH_SPACE["channels"]))
            kernel_sizes.append(trial.suggest_categorical(f"k_b{i}", SEARCH_SPACE["kernel_sizes"]))
            block_types.append(trial.suggest_categorical(f"type_b{i}", SEARCH_SPACE["block_types"]))
            activations.append(trial.suggest_categorical(f"act_b{i}", SEARCH_SPACE["activations"]))

        return {
            "num_blocks": num_blocks,
            "input_resolution": input_resolution,
            "channels": channels,
            "kernel_sizes": kernel_sizes,
            "block_types": block_types,
            "activations": activations,
        }

    @staticmethod
    def build_model(config: Dict[str, Any], num_classes: int = 10) -> SearchableCNN:
        return SearchableCNN(config=config, num_classes=num_classes)
