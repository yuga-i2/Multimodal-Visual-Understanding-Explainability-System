"""Evaluation metrics for classification and segmentation tasks.

Pure PyTorch implementations of standard metrics:
- Classification: Accuracy, Top-k Accuracy
- Segmentation: Dice coefficient, IoU (Jaccard index)
"""

import torch


class ClassificationMetrics:
    """Accumulates and computes classification metrics."""

    def __init__(self) -> None:
        """Initialize metric accumulators."""
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators."""
        self.total_correct = 0
        self.total_top_k_correct = 0
        self.total_samples = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> None:
        """Update metrics with batch predictions.

        Args:
            outputs: Model outputs (logits) [B, C] where C is number of classes.
            targets: Ground truth labels [B] (long tensor).
            k: For top-k accuracy (default 5).

        Raises:
            ValueError: If outputs and targets have incompatible shapes.
        """
        if outputs.size(0) != targets.size(0):
            raise ValueError(
                f"Batch size mismatch: outputs {outputs.size(0)} vs targets {targets.size(0)}"
            )

        batch_size = outputs.size(0)

        # Accuracy
        predictions = outputs.argmax(dim=1)
        correct = (predictions == targets).sum().item()
        self.total_correct += correct

        # Top-k accuracy
        num_classes = outputs.size(1)
        k_actual = min(k, num_classes)
        _, top_k_preds = outputs.topk(k_actual, dim=1)
        top_k_correct = top_k_preds.eq(targets.unsqueeze(1).expand_as(top_k_preds)).any(dim=1).sum().item()
        self.total_top_k_correct += top_k_correct

        self.total_samples += batch_size

    def compute(self) -> dict:
        """Compute accumulated metrics.

        Returns:
            Dictionary with keys:
            - 'accuracy': Classification accuracy (0 to 1)
            - 'top_5_accuracy': Top-5 accuracy (0 to 1)
        """
        if self.total_samples == 0:
            return {"accuracy": 0.0, "top_5_accuracy": 0.0}

        return {
            "accuracy": self.total_correct / self.total_samples,
            "top_5_accuracy": self.total_top_k_correct / self.total_samples,
        }


class SegmentationMetrics:
    """Accumulates and computes segmentation metrics."""

    def __init__(self, num_classes: int) -> None:
        """Initialize metric accumulators.

        Args:
            num_classes: Number of segmentation classes.
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators."""
        # Per-class IoU: track TP, FP, FN for each class
        self.iou_tp = torch.zeros(self.num_classes)
        self.iou_fp = torch.zeros(self.num_classes)
        self.iou_fn = torch.zeros(self.num_classes)

        # Dice coefficient per class
        self.dice_tp = torch.zeros(self.num_classes)
        self.dice_total_pred = torch.zeros(self.num_classes)
        self.dice_total_target = torch.zeros(self.num_classes)

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics with batch predictions.

        Args:
            outputs: Model outputs (logits) [B, C, H, W].
            targets: Ground truth masks [B, H, W] (long tensor with class indices).

        Raises:
            ValueError: If outputs and targets have incompatible shapes.
        """
        if outputs.size(0) != targets.size(0):
            raise ValueError(
                f"Batch size mismatch: outputs {outputs.size(0)} vs targets {targets.size(0)}"
            )
        if outputs.size(2) != targets.size(1) or outputs.size(3) != targets.size(2):
            raise ValueError(
                f"Spatial size mismatch: outputs {outputs.shape} vs targets {targets.shape}"
            )

        # Get predictions (argmax over class dimension)
        predictions = outputs.argmax(dim=1)  # [B, H, W]

        # Move to CPU for accumulation
        predictions = predictions.cpu()
        targets = targets.cpu()

        # Compute per-class metrics
        for class_idx in range(self.num_classes):
            pred_mask = predictions == class_idx
            target_mask = targets == class_idx

            # IoU components
            tp = (pred_mask & target_mask).sum().item()
            fp = (pred_mask & ~target_mask).sum().item()
            fn = (~pred_mask & target_mask).sum().item()

            self.iou_tp[class_idx] += tp
            self.iou_fp[class_idx] += fp
            self.iou_fn[class_idx] += fn

            # Dice components
            self.dice_tp[class_idx] += tp
            self.dice_total_pred[class_idx] += pred_mask.sum().item()
            self.dice_total_target[class_idx] += target_mask.sum().item()

    def compute(self) -> dict:
        """Compute accumulated metrics.

        Returns:
            Dictionary with keys:
            - 'iou_per_class': Per-class IoU values [num_classes]
            - 'mean_iou': Mean IoU across classes
            - 'dice_per_class': Per-class Dice coefficients [num_classes]
            - 'mean_dice': Mean Dice coefficient across classes
        """
        # Compute per-class IoU
        iou_per_class = torch.zeros(self.num_classes)
        for class_idx in range(self.num_classes):
            denominator = self.iou_tp[class_idx] + self.iou_fp[class_idx] + self.iou_fn[class_idx]
            if denominator > 0:
                iou_per_class[class_idx] = self.iou_tp[class_idx] / denominator
            else:
                iou_per_class[class_idx] = 0.0

        # Compute per-class Dice
        dice_per_class = torch.zeros(self.num_classes)
        for class_idx in range(self.num_classes):
            denominator = self.dice_total_pred[class_idx] + self.dice_total_target[class_idx]
            if denominator > 0:
                dice_per_class[class_idx] = 2.0 * self.dice_tp[class_idx] / denominator
            else:
                dice_per_class[class_idx] = 0.0

        return {
            "iou_per_class": iou_per_class.tolist(),
            "mean_iou": iou_per_class.mean().item(),
            "dice_per_class": dice_per_class.tolist(),
            "mean_dice": dice_per_class.mean().item(),
        }
