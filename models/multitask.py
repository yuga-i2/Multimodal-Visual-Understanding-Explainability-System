"""Multi-task model wrapper combining backbone + attention + task heads.

Flexible architecture that allows:
- Backbone selection (CNN, ViT, or hybrid)
- Optional attention modules
- Classification and/or segmentation heads
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Simple classification head: global pooling + linear."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.5) -> None:
        """Initialize classification head.

        Args:
            in_channels: Number of input channels from backbone.
            num_classes: Number of output classes.
            dropout: Dropout rate.
        """
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature map [B, C, H, W].

        Returns:
            Class logits [B, num_classes].
        """
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SegmentationHead(nn.Module):
    """Simple segmentation head: 1x1 conv + upsample."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        upsampling_factor: int = 4,
    ) -> None:
        """Initialize segmentation head.

        Args:
            in_channels: Number of input channels.
            num_classes: Number of output classes.
            upsampling_factor: Factor to upsample to original resolution.
        """
        super().__init__()
        self.upsampling_factor = upsampling_factor
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature map [B, C, H, W].

        Returns:
            Segmentation logits [B, num_classes, H', W'] where H'=H*factor.
        """
        x = self.conv(x)
        if self.upsampling_factor > 1:
            x = nn.functional.interpolate(
                x,
                scale_factor=self.upsampling_factor,
                mode="bilinear",
                align_corners=False,
            )
        return x


class MultiTaskModel(nn.Module):
    """Unified multi-task model supporting classification and segmentation.

    Architecture:
        Backbone (CNN/ViT/Hybrid)
            |
        Optional Attention (CBAM)
            |
        ├─> Classification Head
        │
        └─> Segmentation Head (optional)
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_out_channels: int,
        num_classes: int,
        include_segmentation_head: bool = False,
        attention_module: Optional[nn.Module] = None,
        seg_upsampling_factor: int = 4,
        classification_dropout: float = 0.5,
    ) -> None:
        """Initialize multi-task model.

        Args:
            backbone: Feature extraction backbone (must output [B, C, H, W] tensors).
            backbone_out_channels: Number of output channels from backbone.
            num_classes: Number of classes for both tasks.
            include_segmentation_head: If True, include segmentation head.
            attention_module: Optional attention module (e.g., CBAM). If provided,
                applied after backbone.
            seg_upsampling_factor: Upsampling factor for segmentation head.
            classification_dropout: Dropout rate in classification head.
        """
        super().__init__()

        self.backbone = backbone
        self.attention_module = attention_module if attention_module is not None else nn.Identity()
        self.include_segmentation_head = include_segmentation_head

        # Classification head
        self.classification_head = ClassificationHead(
            backbone_out_channels,
            num_classes,
            dropout=classification_dropout,
        )

        # Segmentation head (optional)
        if include_segmentation_head:
            self.segmentation_head = SegmentationHead(
                backbone_out_channels,
                num_classes,
                upsampling_factor=seg_upsampling_factor,
            )
        else:
            self.segmentation_head = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input image [B, C, H, W].

        Returns:
            Dictionary with:
            - 'classification': class logits [B, num_classes]
            - 'segmentation': (optional) segmentation logits [B, num_classes, H', W']
        """
        # Backbone forward
        backbone_out = self.backbone(x)

        # Handle different backbone output formats
        if isinstance(backbone_out, list):
            # Multi-scale backbone output: take the deepest feature
            features = backbone_out[-1]
        else:
            features = backbone_out

        # Apply attention (if provided)
        features = self.attention_module(features)

        # Classification head
        logits_cls = self.classification_head(features)

        output = {"classification": logits_cls}

        # Segmentation head (if included)
        if self.include_segmentation_head:
            logits_seg = self.segmentation_head(features)
            output["segmentation"] = logits_seg

        return output

    def forward_classification(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only classification logits.

        Args:
            x: Input image [B, C, H, W].

        Returns:
            Class logits [B, num_classes].
        """
        output = self.forward(x)
        return output["classification"]

    def forward_segmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only segmentation logits.

        Args:
            x: Input image [B, C, H, W].

        Returns:
            Segmentation logits [B, num_classes, H', W'].

        Raises:
            RuntimeError: If segmentation head is not included.
        """
        if not self.include_segmentation_head:
            raise RuntimeError("Segmentation head not included in this model.")

        output = self.forward(x)
        return output["segmentation"]


class HybridModel(nn.Module):
    """CNN-Transformer hybrid model.

    Combines CNN early layers with Transformer for feature extraction.
    Useful for architectures that benefit from both local and global context.
    """

    def __init__(
        self,
        cnn_backbone: nn.Module,
        transformer_backbone: nn.Module,
        cnn_output_idx: int = 2,
        num_classes: int = 10,
        include_segmentation_head: bool = False,
    ) -> None:
        """Initialize hybrid model.

        Args:
            cnn_backbone: CNN backbone (should return list of features).
            transformer_backbone: Transformer backbone that accepts [B, C, H, W].
            cnn_output_idx: Which CNN stage output to use before transformer.
            num_classes: Number of classes.
            include_segmentation_head: If True, include segmentation head.
        """
        super().__init__()
        self.cnn_backbone = cnn_backbone
        self.transformer_backbone = transformer_backbone
        self.cnn_output_idx = cnn_output_idx
        self.include_segmentation_head = include_segmentation_head

        # Estimate output channels from transformer (typical ViT values)
        transformer_channels = getattr(transformer_backbone, "embed_dim", 768)

        self.classification_head = ClassificationHead(transformer_channels, num_classes)

        if include_segmentation_head:
            self.segmentation_head = SegmentationHead(transformer_channels, num_classes)
        else:
            self.segmentation_head = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through hybrid architecture.

        Args:
            x: Input image [B, C, H, W].

        Returns:
            Dictionary with classification (and optionally segmentation) outputs.
        """
        # CNN forward: get features at specified stage
        cnn_features = self.cnn_backbone(x)
        cnn_out = cnn_features[self.cnn_output_idx]

        # Transformer forward: process CNN output
        transformer_out = self.transformer_backbone.forward_spatial(cnn_out)

        # Classification
        logits_cls = self.classification_head(transformer_out)

        output = {"classification": logits_cls}

        # Segmentation (if included)
        if self.include_segmentation_head:
            logits_seg = self.segmentation_head(transformer_out)
            output["segmentation"] = logits_seg

        return output
