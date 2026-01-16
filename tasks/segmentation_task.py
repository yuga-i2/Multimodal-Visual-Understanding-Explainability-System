"""Segmentation task wrapper.

Wraps a multi-task or segmentation model for dense prediction tasks.
Provides standardized interface for segmentation workflows.

This is a minimal Phase 1 wrapper. Full training logic, metrics,
and loss computation will be added in later phases.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationTask(nn.Module):
    """Wrapper for segmentation tasks.

    Encapsulates segmentation model and provides standardized interface
    for segmentation workflows (inference, prediction, confidence).
    """

    def __init__(self, model: nn.Module, num_classes: int) -> None:
        """Initialize segmentation task.

        Args:
            model: Segmentation model (outputs logits [B, C, H, W]).
            num_classes: Number of segmentation classes.
        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation.

        Args:
            x: Input image [B, C, H, W].

        Returns:
            Segmentation logits [B, num_classes, H', W'].
        """
        # If model returns dict (multi-task), extract segmentation logits
        output = self.model(x)
        if isinstance(output, dict):
            return output["segmentation"]
        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get segmentation predictions (class indices).

        Args:
            x: Input image [B, C, H, W].

        Returns:
            Predicted class mask [B, H', W'].
        """
        logits = self.forward(x)
        return logits.argmax(dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get per-pixel class probabilities.

        Args:
            x: Input image [B, C, H, W].

        Returns:
            Class probabilities [B, num_classes, H', W'].
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Get confidence of predictions (max probability).

        Args:
            x: Input image [B, C, H, W].

        Returns:
            Confidence scores [B, H', W'] (values in [0, 1]).
        """
        proba = self.predict_proba(x)
        confidence = proba.max(dim=1)[0]
        return confidence
