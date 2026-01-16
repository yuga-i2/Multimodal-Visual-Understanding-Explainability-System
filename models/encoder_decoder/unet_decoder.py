"""U-Net encoder-decoder for segmentation.

Generic encoder-decoder architecture that accepts arbitrary backbone feature
maps and produces dense predictions. Supports variable-depth architectures.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Decoder block: upsample + conv + optional skip connection."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        """Initialize decoder block.

        Args:
            in_channels: Channels from deeper layer.
            skip_channels: Channels from skip connection (0 if no skip).
            out_channels: Output channels.
        """
        super().__init__()

        # Upsample to match skip resolution (if skip exists)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # After upsample + skip concatenation
        combined_channels = in_channels + skip_channels

        # Convolution to reduce channels and refine features
        self.conv_block = nn.Sequential(
            nn.Conv2d(combined_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional skip connection.

        Args:
            x: Input feature map [B, in_channels, H, W].
            skip: Optional skip connection [B, skip_channels, 2*H, 2*W].

        Returns:
            Decoded features [B, out_channels, 2*H, 2*W].
        """
        x = self.upsample(x)

        if skip is not None:
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)

        x = self.conv_block(x)
        return x


class UNetDecoder(nn.Module):
    """U-Net decoder that accepts encoder features.

    Progressively upsamples encoder features and combines them with skip
    connections. Designed to work with arbitrary backbones (CNN, ViT, etc.).
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        num_classes: int,
    ) -> None:
        """Initialize U-Net decoder.

        Args:
            encoder_channels: List of encoder feature map channels.
                For example, a 4-level encoder outputs [64, 128, 256, 512].
                Must be in increasing order (shallow to deep).
            decoder_channels: List of decoder channels at each level.
                Should have length = len(encoder_channels) - 1.
            num_classes: Number of output classes.
        """
        super().__init__()

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.num_classes = num_classes

        # Build decoder blocks
        # Decoder starts from deepest encoder feature and progressively upsamples
        self.decoder_blocks = nn.ModuleList()

        # Starting from the deepest encoder output, work backwards
        for i in range(len(decoder_channels)):
            # Current layer: receiving features from (i + 1) position in encoder
            in_ch = encoder_channels[-(i + 1)]

            # Skip connection from one level up
            skip_ch = encoder_channels[-(i + 2)] if i < len(encoder_channels) - 1 else 0

            # Output channels for this decoder level
            out_ch = decoder_channels[i]

            block = DecoderBlock(in_ch, skip_ch, out_ch)
            self.decoder_blocks.append(block)

        # Final segmentation head
        final_channels = decoder_channels[-1]
        self.seg_head = nn.Sequential(
            nn.Conv2d(final_channels, final_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels, num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        encoder_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass through decoder.

        Args:
            encoder_features: List of feature maps from encoder, in order
                from shallow to deep. Length should equal len(encoder_channels).

        Returns:
            Logits of shape [B, num_classes, H_orig, W_orig].
        """
        # Start from the deepest encoder feature
        x = encoder_features[-1]

        # Progressively decode, incorporating skip connections
        for i, block in enumerate(self.decoder_blocks):
            # Determine skip connection (if not the last block)
            skip_idx = -(i + 2) if i < len(encoder_features) - 1 else None
            skip = encoder_features[skip_idx] if skip_idx is not None else None

            x = block(x, skip)

        # Final segmentation head
        logits = self.seg_head(x)

        return logits


class UNet(nn.Module):
    """Full U-Net segmentation model.

    Combines a backbone encoder with a U-Net decoder. The backbone
    must return a list of feature maps at progressively deeper levels.
    """

    def __init__(
        self,
        backbone: nn.Module,
        encoder_channels: List[int],
        decoder_channels: List[int],
        num_classes: int,
    ) -> None:
        """Initialize U-Net.

        Args:
            backbone: Feature extraction backbone that returns list of feature maps.
            encoder_channels: Channels at each backbone stage.
            decoder_channels: Channels at each decoder stage.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.backbone = backbone
        self.decoder = UNetDecoder(encoder_channels, decoder_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image [B, C, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # Encoder: extract multi-scale features
        encoder_features = self.backbone(x)

        # Decoder: combine features and produce segmentation
        logits = self.decoder(encoder_features)

        return logits


def create_unet(
    backbone: nn.Module,
    encoder_channels: List[int],
    num_classes: int,
    decoder_channels: Optional[List[int]] = None,
) -> UNet:
    """Factory function to create U-Net with default decoder configuration.

    Args:
        backbone: Feature extraction backbone.
        encoder_channels: Channel counts from backbone.
        num_classes: Number of segmentation classes.
        decoder_channels: Optional explicit decoder channels. If None, uses
            encoder_channels in reverse order (standard U-Net).

    Returns:
        Initialized UNet model.
    """
    if decoder_channels is None:
        # Use encoder channels in reverse for symmetric architecture
        decoder_channels = list(reversed(encoder_channels[:-1]))

    return UNet(
        backbone,
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        num_classes=num_classes,
    )
