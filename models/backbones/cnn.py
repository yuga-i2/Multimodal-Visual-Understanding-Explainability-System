"""Simple CNN backbone for feature extraction.

Provides a headless feature extractor with ResNet-style residual blocks.
Returns spatial feature maps suitable for downstream tasks (classification, segmentation).
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic Conv -> BatchNorm -> ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize convolutional block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for convolution.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for downsampling (1 = no downsampling).
        """
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        # Shortcut: 1x1 conv with stride if dimensions change
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out


class CNNBackbone(nn.Module):
    """Headless CNN backbone for feature extraction.

    Multi-scale feature extractor using residual blocks.
    Returns intermediate feature maps at different spatial resolutions.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 4,
    ) -> None:
        """Initialize CNN backbone.

        Args:
            in_channels: Number of input channels (default 3 for RGB).
            base_channels: Number of channels in first layer.
            num_blocks: Number of residual blocks per stage.
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        # Initial conv layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages with increasing channel depth
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        strides = [1, 2, 2, 2]

        self.stages = nn.ModuleList()
        for i, (ch, stride) in enumerate(zip(channels, strides)):
            prev_ch = base_channels if i == 0 else channels[i - 1]
            blocks = nn.Sequential(
                ResidualBlock(prev_ch, ch, stride=stride),
                *[ResidualBlock(ch, ch, stride=1) for _ in range(num_blocks - 1)],
            )
            self.stages.append(blocks)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning multi-scale feature maps.

        Args:
            x: Input tensor with shape [B, C, H, W].

        Returns:
            List of feature maps at different scales:
            - stem output: [B, base_channels, H/4, W/4]
            - stage1-4: progressively deeper and smaller
        """
        features = []

        # Stem: reduce spatial size, increase channels
        x = self.stem(x)
        features.append(x)

        # Stages: residual blocks with downsampling
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only the deepest feature map.

        Useful for classification tasks that don't need multi-scale features.

        Args:
            x: Input tensor with shape [B, C, H, W].

        Returns:
            Deepest feature map [B, base_channels*8, H/32, W/32].
        """
        features = self.forward(x)
        return features[-1]


def create_cnn_backbone(
    in_channels: int = 3,
    base_channels: int = 64,
    depth: str = "small",
) -> CNNBackbone:
    """Factory function to create CNN backbone with preset configurations.

    Args:
        in_channels: Number of input channels.
        base_channels: Base channel count.
        depth: Preset depth ("small", "medium", "large").
               - small: 2 blocks per stage
               - medium: 3 blocks per stage
               - large: 4 blocks per stage

    Returns:
        Initialized CNNBackbone.
    """
    num_blocks = {"small": 2, "medium": 3, "large": 4}[depth]
    return CNNBackbone(
        in_channels=in_channels,
        base_channels=base_channels,
        num_blocks=num_blocks,
    )
