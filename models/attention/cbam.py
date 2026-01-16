"""Convolutional Block Attention Module (CBAM).

CBAM applies attention along channel and spatial dimensions, enhancing
feature maps by suppressing less relevant information.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention module."""

    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        """Initialize channel attention.

        Args:
            channels: Number of input channels.
            reduction_ratio: Reduction factor for bottleneck dimension.
        """
        super().__init__()
        reduced_channels = max(channels // reduction_ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Attention-weighted tensor [B, C, H, W].
        """
        # Average pooling branch
        avg_feat = self.avg_pool(x)
        avg_feat = self.mlp(avg_feat)

        # Max pooling branch
        max_feat = self.max_pool(x)
        max_feat = self.mlp(max_feat)

        # Combine branches
        channel_attn = self.sigmoid(avg_feat + max_feat)
        return x * channel_attn


class SpatialAttention(nn.Module):
    """Spatial attention module."""

    def __init__(self, kernel_size: int = 7) -> None:
        """Initialize spatial attention.

        Args:
            kernel_size: Kernel size for conv (must be odd).
        """
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Attention-weighted tensor [B, C, H, W].
        """
        # Channel-wise statistics
        avg_feat = torch.mean(x, dim=1, keepdim=True)
        max_feat = torch.max(x, dim=1, keepdim=True)[0]

        # Concatenate statistics: [B, 2, H, W]
        feat = torch.cat([avg_feat, max_feat], dim=1)

        # Apply convolution + sigmoid
        spatial_attn = self.sigmoid(self.conv(feat))
        return x * spatial_attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Sequentially applies channel attention and spatial attention to refine
    feature maps by suppressing irrelevant channels and spatial regions.

    Reference:
        Woo et al. "CBAM: Convolutional Block Attention Module" ECCV 2018.
    """

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
    ) -> None:
        """Initialize CBAM.

        Args:
            channels: Number of input channels.
            reduction_ratio: Reduction factor for channel attention MLP.
            spatial_kernel_size: Kernel size for spatial attention.
        """
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction_ratio)
        self.spatial_attn = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel and spatial attention sequentially.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Attention-refined tensor [B, C, H, W].
        """
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class OptionalCBAM(nn.Module):
    """Wrapper to optionally apply CBAM based on a flag.

    Useful for making attention modules optional during architecture design,
    allowing easy ablation studies without code changes.
    """

    def __init__(self, channels: int, enable: bool = True, **kwargs) -> None:
        """Initialize optional CBAM.

        Args:
            channels: Number of input channels.
            enable: If False, becomes identity operation.
            **kwargs: Additional arguments passed to CBAM.
        """
        super().__init__()
        self.enable = enable
        if enable:
            self.cbam = CBAM(channels, **kwargs)
        else:
            self.cbam = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Attention-refined or unchanged tensor [B, C, H, W].
        """
        return self.cbam(x)
