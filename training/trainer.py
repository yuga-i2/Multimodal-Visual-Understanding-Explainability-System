"""Training loop management and trainer utilities.

This module provides the core training loop implementation for Phase 1 models,
including support for classification, segmentation, and multi-task learning
with mixed precision, gradient scaling, and checkpoint management.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .mixed_precision import MixedPrecisionManager


class Trainer:
    """Unified PyTorch training engine for Phase 1 models.
    
    Supports:
    - Classification (outputs: logits; targets: {"label": int})
    - Segmentation (outputs: logits; targets: {"mask": [H, W]})
    - Multi-task (outputs: {"classification": ..., "segmentation": ...})
    
    Uses dict-based sample format from data/ module. Requires loss functions
    compatible with model outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Union[Callable, Dict[str, Callable]],
        device: str = "cpu",
        amp_enabled: bool = True,
        checkpoint_dir: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Phase 1 model (supports any nn.Module).
            optimizer: Optimizer for model parameters.
            loss_fn: Loss function or dict of loss functions for multi-task.
            device: Device to run training on ("cpu" or "cuda").
            amp_enabled: Enable automatic mixed precision for GPU training.
            checkpoint_dir: Directory to save checkpoints.
            seed: Random seed for reproducibility.

        Raises:
            RuntimeError: If device is invalid.
        """
        # Validate device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.amp_manager = MixedPrecisionManager(
            enabled=amp_enabled and device == "cuda"
        )
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)

        # Training state
        self.epoch = 0
        self.best_loss = float("inf")
        self.train_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        # Create checkpoint directory if needed
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)


    def _compute_loss(
        self,
        model_output: Union[torch.Tensor, Dict[str, torch.Tensor]],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss based on model output and targets.

        Handles:
        - Classification: outputs [B, C], targets["label"] -> CrossEntropyLoss
        - Segmentation: outputs [B, C, H, W], targets["mask"] -> CrossEntropyLoss
        - Multi-task: outputs dict -> weighted sum of individual losses

        Args:
            model_output: Model output (tensor or dict of tensors).
            batch: Batch dict with "image" and target keys ("label" or "mask").

        Returns:
            Scalar loss tensor.
        """
        # Multi-task: loss dict provided, outputs dict expected
        if isinstance(self.loss_fn, dict):
            total_loss = torch.tensor(0.0, device=self.device)
            
            if isinstance(model_output, dict):
                # Multi-task model: {"classification": ..., "segmentation": ...}
                if "classification" in model_output and "classification" in self.loss_fn:
                    cls_loss = self.loss_fn["classification"](
                        model_output["classification"], 
                        batch["label"]
                    )
                    total_loss = total_loss + cls_loss
                
                if "segmentation" in model_output and "segmentation" in self.loss_fn:
                    seg_loss = self.loss_fn["segmentation"](
                        model_output["segmentation"],
                        batch["mask"]
                    )
                    total_loss = total_loss + seg_loss
            
            return total_loss
        
        # Single-task: single loss function
        if isinstance(model_output, dict):
            # Model output dict but single loss - shouldn't happen in normal use
            raise ValueError(
                "Model output is dict but loss_fn is not. "
                "Pass loss_fn as dict for multi-task models."
            )
        
        # Classification or Segmentation
        if "label" in batch:
            # Classification
            return self.loss_fn(model_output, batch["label"])
        elif "mask" in batch:
            # Segmentation
            return self.loss_fn(model_output, batch["mask"])
        else:
            raise ValueError("Batch must contain 'label' or 'mask' key")

    def train_one_epoch(self, train_loader: DataLoader) -> float:
        """Run a single training epoch.

        Args:
            train_loader: DataLoader for training data (yields dict samples).

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with self.amp_manager.get_autocast_context():
                model_output = self.model(batch["image"])
                loss = self._compute_loss(model_output, batch)

            # Backward pass with AMP scaling
            scaled_loss = self.amp_manager.scale_loss(loss)
            scaled_loss.backward()

            # Gradient clipping (unscale first if AMP enabled)
            if self.amp_manager.enabled:
                self.amp_manager.unscale_gradients(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            if self.amp_manager.enabled:
                self.amp_manager.step_scaler(self.optimizer)
            else:
                self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else float("inf")

    def validate(self, val_loader: DataLoader) -> float:
        """Run validation on validation set.

        Args:
            val_loader: DataLoader for validation data (yields dict samples).

        Returns:
            Average loss for validation set.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                with self.amp_manager.get_autocast_context():
                    model_output = self.model(batch["image"])
                    loss = self._compute_loss(model_output, batch)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else float("inf")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        patience: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Fit the model on training data.

        Args:
            train_loader: DataLoader for training data (yields dict samples).
            val_loader: Optional DataLoader for validation data.
            epochs: Number of epochs to train.
            patience: Early stopping patience (number of epochs without improvement).

        Returns:
            Training history dict with "train_loss" and "val_loss" keys.
        """
        early_stop_counter = 0

        for epoch in range(epochs):
            self.epoch = epoch

            # Training phase
            train_loss = self.train_one_epoch(train_loader)
            self.train_history["train_loss"].append(train_loss)

            # Validation phase
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.train_history["val_loss"].append(val_loss)

                # Early stopping
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    early_stop_counter = 0
                    # Auto-save best model
                    if self.checkpoint_dir:
                        self.save_checkpoint(
                            os.path.join(self.checkpoint_dir, "best_model.pt")
                        )
                else:
                    early_stop_counter += 1
                    if patience is not None and early_stop_counter >= patience:
                        print(
                            f"Early stopping at epoch {epoch+1}. "
                            f"Best val_loss: {self.best_loss:.6f}"
                        )
                        break

                print(
                    f"Epoch [{epoch+1}/{epochs}] - "
                    f"train_loss: {train_loss:.6f} | "
                    f"val_loss: {val_loss:.6f}"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{epochs}] - "
                    f"train_loss: {train_loss:.6f}"
                )

        return self.train_history

    @staticmethod
    def _default_logger(message: str) -> None:
        """Default logger (prints to console).

        Args:
            message: Message to log.
        """
        print(message)


    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint.

        Saves model state dict, optimizer state, and training history.
        Minimal checkpoint (just model state, no training state yet).

        Args:
            filepath: Path to save checkpoint.
        """
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "train_history": self.train_history,
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint.

        Args:
            filepath: Path to checkpoint file.

        Raises:
            FileNotFoundError: If checkpoint file not found.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.train_history = checkpoint.get("train_history", self.train_history)

        print(f"Checkpoint loaded from {filepath}")

    def get_model(self) -> nn.Module:
        """Get the trained model.

        Returns:
            Model instance.
        """
        return self.model

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history.

        Returns:
            Dictionary with training metrics over epochs.
        """
        return self.train_history
