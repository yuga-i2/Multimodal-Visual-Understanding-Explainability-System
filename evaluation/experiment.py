"""Experiment orchestrator for training and evaluation.

Coordinates Trainer (Phase 2) and Evaluator (Phase 3) without modification.
"""

from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from evaluation.reproducibility import set_seed


class Experiment:
    """Lightweight orchestrator for training and evaluation.
    
    Coordinates Phase 2 Trainer with Phase 3 Evaluator to run complete
    experiments without hyperparameter search.
    """

    def __init__(
        self,
        model: nn.Module,
        trainer: Trainer,
        evaluator: Evaluator,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize experiment.

        Args:
            model: Phase 1 model (any architecture).
            trainer: Phase 2 Trainer instance (configured and ready).
            evaluator: Phase 3 Evaluator instance (with device and model set).
            seed: Random seed for reproducibility. If None, no seed is set.
        """
        self.model = model
        self.trainer = trainer
        self.evaluator = evaluator
        self.seed = seed

        # Set seed if provided
        if seed is not None:
            set_seed(seed)

        # History tracking
        self.train_history: Dict[str, List[float]] = {
            "loss": [],
            "task_losses": {},
        }
        self.val_history: Dict[str, Union[List[float], List[Dict]]] = {
            "metrics": [],
        }

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        task: str = "auto",
        num_classes: Optional[int] = None,
        num_classes_segmentation: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run full experiment (training + evaluation).

        Args:
            train_loader: Training DataLoader (Phase 2 format).
            val_loader: Validation DataLoader (Phase 2 format).
            num_epochs: Number of training epochs.
            task: Task type ("classification", "segmentation", "multi-task", or "auto").
            num_classes: Number of classes (required for segmentation/multi-task).
            num_classes_segmentation: Number of segmentation classes (multi-task only).

        Returns:
            Dictionary with complete experiment results:
            - 'train_loss': List of training losses per epoch
            - 'val_metrics': List of validation metric dicts per epoch
            - 'best_epoch': Index of best validation epoch
            - 'best_metrics': Best validation metrics
            - 'final_model_state': Final model state dict (optional)
        """
        best_epoch = 0
        best_val_metric = None
        results = {
            "train_loss": [],
            "val_metrics": [],
            "best_epoch": -1,
            "best_metrics": {},
        }

        for epoch in range(num_epochs):
            # Training phase
            train_loss = self.trainer.train_one_epoch(train_loader)
            results["train_loss"].append(train_loss)
            self.train_history["loss"].append(train_loss)

            # Validation phase
            val_metrics = self.evaluator.evaluate(
                val_loader,
                task=task,
                num_classes=num_classes,
                num_classes_segmentation=num_classes_segmentation,
            )
            results["val_metrics"].append(val_metrics)
            self.val_history["metrics"].append(val_metrics)

            # Track best metrics (using first metric key as primary)
            metric_values = list(val_metrics.values())
            if metric_values:
                current_metric = (
                    metric_values[0]
                    if isinstance(metric_values[0], (int, float))
                    else 0.0
                )
                if best_val_metric is None or current_metric > best_val_metric:
                    best_val_metric = current_metric
                    best_epoch = epoch
                    results["best_epoch"] = best_epoch
                    results["best_metrics"] = val_metrics.copy()

            # Optional: Print progress
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Metrics: {val_metrics}"
            )

        return results

    def run_validation_only(
        self,
        val_loader: DataLoader,
        task: str = "auto",
        num_classes: Optional[int] = None,
        num_classes_segmentation: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run validation only (no training).

        Args:
            val_loader: Validation DataLoader.
            task: Task type ("classification", "segmentation", "multi-task", or "auto").
            num_classes: Number of classes (required for segmentation/multi-task).
            num_classes_segmentation: Number of segmentation classes (multi-task only).

        Returns:
            Dictionary with validation metrics.
        """
        val_metrics = self.evaluator.evaluate(
            val_loader,
            task=task,
            num_classes=num_classes,
            num_classes_segmentation=num_classes_segmentation,
        )

        self.val_history["metrics"].append(val_metrics)

        return {
            "val_metrics": val_metrics,
        }

    def get_history(self) -> Dict[str, Any]:
        """Get training and validation history.

        Returns:
            Dictionary with:
            - 'train_loss': List of training losses per epoch
            - 'val_metrics': List of validation metric dicts per epoch
        """
        return {
            "train_loss": self.train_history["loss"],
            "val_metrics": self.val_history["metrics"],
        }

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save experiment checkpoint including model state and history.

        Args:
            checkpoint_path: Path to save checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "train_history": self.train_history,
            "val_history": self.val_history,
            "seed": self.seed,
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load experiment checkpoint.

        Args:
            checkpoint_path: Path to load checkpoint from.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.train_history = checkpoint.get("train_history", {})
        self.val_history = checkpoint.get("val_history", {})
        if checkpoint.get("seed") is not None:
            self.seed = checkpoint["seed"]
