"""Classification task wrapper.

Wraps a multi-task model for classification-only inference and training.
Provides a clean interface for classification experiments.
"""

from typing import Optional
import torch
import torch.nn as nn


class ClassificationTask(nn.Module):
    """Wrapper for classification tasks.

    Encapsulates model and provides standardized interface for
    classification workflows (training, validation, inference).

    This is a minimal Phase 1 wrapper. Full training logic, metrics,
    and loss computation will be added in later phases.
    """

    def __init__(self, model: nn.Module, num_classes: int) -> None:
        """Initialize classification task.

        Args:
            model: Multi-task model or backbone with classification head.
            num_classes: Number of classification classes.
        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification.

        Args:
            x: Input image [B, C, H, W].

        Returns:
            Class logits [B, num_classes].
        """
        # If model returns dict (multi-task), extract classification logits
        output = self.model(x)
        if isinstance(output, dict):
            return output["classification"]
        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions and confidence scores.

        Args:
            x: Input image [B, C, H, W].

        Returns:
            Predicted class indices [B].
        """
        logits = self.forward(x)
        return logits.argmax(dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities.

        Args:
            x: Input image [B, C, H, W].

        Returns:
            Class probabilities [B, num_classes].
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
