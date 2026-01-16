"""U-Net: Convolutional Networks for Biomedical Image Segmentation.

Reference: Ronneberger et al., 2015 - https://arxiv.org/abs/1505.04597
"""

from typing import Optional
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Double convolution block: Conv2D -> ReLU -> Conv2D -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize double convolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.double_conv(x)


class UpConv(nn.Module):
    """Upsampling followed by convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize upsampling convolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=2,
                padding=0,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Upsampled and convolved tensor.
        """
        return self.up_conv(x)


class UNet(nn.Module):
    """U-Net architecture for semantic segmentation.

    Features:
    - Symmetric encoder-decoder structure
    - Skip connections for feature preservation
    - Configurable input channels, number of classes, and base channels
    - Dropout regularization in bottleneck
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 64,
    ) -> None:
        """Initialize U-Net.

        Args:
            input_channels: Number of input channels (default: 1 for grayscale).
            num_classes: Number of output classes (default: 1 for binary segmentation).
            base_channels: Base number of channels (default: 64). Each level doubles/halves this.
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        # Encoder (downsampling path)
        self.enc1 = DoubleConv(input_channels, base_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        self.dropout = nn.Dropout(p=0.5)

        # Decoder (upsampling path)
        self.upconv4 = UpConv(base_channels * 16, base_channels * 8)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.upconv3 = UpConv(base_channels * 8, base_channels * 4)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.upconv2 = UpConv(base_channels * 4, base_channels * 2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.upconv1 = UpConv(base_channels * 2, base_channels)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Output layer
        self.final_conv = nn.Conv2d(
            base_channels,
            num_classes,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Output segmentation map of shape (batch, num_classes, height, width).
        """
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(pool4)
        bottleneck = self.dropout(bottleneck)

        # Decoder
        upconv4 = self.upconv4(bottleneck)
        merge4 = torch.cat([enc4, upconv4], dim=1)
        dec4 = self.dec4(merge4)

        upconv3 = self.upconv3(dec4)
        merge3 = torch.cat([enc3, upconv3], dim=1)
        dec3 = self.dec3(merge3)

        upconv2 = self.upconv2(dec3)
        merge2 = torch.cat([enc2, upconv2], dim=1)
        dec2 = self.dec2(merge2)

        upconv1 = self.upconv1(dec2)
        merge1 = torch.cat([enc1, upconv1], dim=1)
        dec1 = self.dec1(merge1)

        # Output
        output = self.final_conv(dec1)

        return output


def create_unet(
    input_channels: int = 1,
    num_classes: int = 1,
    base_channels: int = 64,
    pretrained_weights: Optional[str] = None,
) -> UNet:
    """Factory function to create a U-Net model.

    Args:
        input_channels: Number of input channels.
        num_classes: Number of output classes.
        base_channels: Base number of channels in the network.
        pretrained_weights: Optional path to pretrained weights.

    Returns:
        Initialized UNet model.
    """
    model = UNet(
        input_channels=input_channels,
        num_classes=num_classes,
        base_channels=base_channels,
    )

    if pretrained_weights is not None:
        checkpoint = torch.load(pretrained_weights, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    return model
