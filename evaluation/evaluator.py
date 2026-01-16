"""Evaluator for computing metrics on trained models.

Handles evaluation of classification, segmentation, and multi-task models
without modifying the Trainer or models.
"""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import ClassificationMetrics, SegmentationMetrics


class Evaluator:
    """Generic evaluator for any Phase 1 model.
    
    Computes metrics over a full dataset for classification, segmentation,
    or multi-task models. Works with Phase 2 DataLoader producing dict batches.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ) -> None:
        """Initialize evaluator.

        Args:
            model: Trained model (any Phase 1 architecture).
            device: Device for evaluation ("cpu" or "cuda").
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()

    def evaluate_classification(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate classification model.

        Args:
            val_loader: DataLoader yielding dict batches with "image" and "label".

        Returns:
            Dictionary with metrics:
            - 'accuracy': Classification accuracy
            - 'top_5_accuracy': Top-5 accuracy
        """
        metrics = ClassificationMetrics()

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Update metrics
                metrics.update(outputs, labels)

        return metrics.compute()

    def evaluate_segmentation(
        self,
        val_loader: DataLoader,
        num_classes: int,
    ) -> Dict[str, Union[float, list]]:
        """Evaluate segmentation model.

        Args:
            val_loader: DataLoader yielding dict batches with "image" and "mask".
            num_classes: Number of segmentation classes.

        Returns:
            Dictionary with metrics:
            - 'mean_iou': Mean Intersection over Union
            - 'iou_per_class': Per-class IoU values
            - 'mean_dice': Mean Dice coefficient
            - 'dice_per_class': Per-class Dice values
        """
        metrics = SegmentationMetrics(num_classes)

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Update metrics
                metrics.update(outputs, masks)

        return metrics.compute()

    def evaluate_multitask(
        self,
        val_loader: DataLoader,
        num_classes_classification: int,
        num_classes_segmentation: int,
    ) -> Dict[str, Union[float, list]]:
        """Evaluate multi-task model.

        Args:
            val_loader: DataLoader yielding dict batches with "image", "label", "mask".
            num_classes_classification: Number of classification classes.
            num_classes_segmentation: Number of segmentation classes.

        Returns:
            Dictionary with metrics for both tasks:
            - 'classification_accuracy': Classification accuracy
            - 'classification_top_5_accuracy': Top-5 accuracy
            - 'segmentation_mean_iou': Mean IoU
            - 'segmentation_iou_per_class': Per-class IoU
            - 'segmentation_mean_dice': Mean Dice
            - 'segmentation_dice_per_class': Per-class Dice
        """
        cls_metrics = ClassificationMetrics()
        seg_metrics = SegmentationMetrics(num_classes_segmentation)

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                masks = batch["mask"].to(self.device)

                # Forward pass (expects dict output)
                outputs = self.model(images)

                # Update metrics for both tasks
                if isinstance(outputs, dict):
                    # Multi-task model with dict output
                    if "classification" in outputs:
                        cls_metrics.update(outputs["classification"], labels)
                    if "segmentation" in outputs:
                        seg_metrics.update(outputs["segmentation"], masks)
                else:
                    # Single-task model, evaluate on both (unusual but supported)
                    cls_metrics.update(outputs, labels)
                    seg_metrics.update(outputs, masks)

        # Combine results
        results = {}
        cls_results = cls_metrics.compute()
        seg_results = seg_metrics.compute()

        for key, val in cls_results.items():
            results[f"classification_{key}"] = val

        for key, val in seg_results.items():
            results[f"segmentation_{key}"] = val

        return results

    def evaluate(
        self,
        val_loader: DataLoader,
        task: str = "auto",
        num_classes: Optional[int] = None,
        num_classes_segmentation: Optional[int] = None,
    ) -> Dict[str, Union[float, list]]:
        """Generic evaluate method that auto-detects task type.

        Args:
            val_loader: DataLoader with dict batches.
            task: Task type ("classification", "segmentation", "multi-task", or "auto").
            num_classes: Number of classes (required for segmentation/multi-task).
            num_classes_segmentation: Number of segmentation classes (multi-task only).

        Returns:
            Dictionary with computed metrics.

        Raises:
            ValueError: If task cannot be determined or required args missing.
        """
        if task == "auto":
            # Try to infer task from first batch
            first_batch = next(iter(val_loader))
            if "mask" in first_batch and "label" in first_batch:
                task = "multi-task"
            elif "mask" in first_batch:
                task = "segmentation"
            elif "label" in first_batch:
                task = "classification"
            else:
                raise ValueError(
                    "Cannot infer task from batch keys. Provide explicit task argument."
                )

        if task == "classification":
            return self.evaluate_classification(val_loader)

        elif task == "segmentation":
            if num_classes is None:
                raise ValueError("num_classes required for segmentation task")
            return self.evaluate_segmentation(val_loader, num_classes)

        elif task == "multi-task":
            if num_classes is None:
                raise ValueError("num_classes required for multi-task classification")
            if num_classes_segmentation is None:
                raise ValueError("num_classes_segmentation required for multi-task")
            return self.evaluate_multitask(
                val_loader,
                num_classes,
                num_classes_segmentation,
            )

        else:
            raise ValueError(f"Unknown task: {task}")
