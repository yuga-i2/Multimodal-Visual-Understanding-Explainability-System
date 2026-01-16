"""Mixed precision training utilities for memory-efficient training.

This module provides utilities for automatic mixed precision (AMP) training,
gradient scaling, and loss functions compatible with mixed precision.
"""

from contextlib import contextmanager
from typing import Any, Generator, Optional, Union
import torch
from torch.cuda.amp import autocast, GradScaler


@contextmanager
def get_autocast_context(
    enabled: bool = True,
    dtype: str = "float16",
) -> Generator[None, None, None]:
    """Get autocast context manager for mixed precision.

    Args:
        enabled: Whether to enable autocast.
        dtype: Data type for autocast ('float16' or 'bfloat16').

    Yields:
        Autocast context.
    """
    if enabled and torch.cuda.is_available():
        with autocast(dtype=torch.float16 if dtype == "float16" else torch.bfloat16):
            yield
    else:
        yield


def get_grad_scaler(enabled: bool = True) -> Union[GradScaler, object]:
    """Get gradient scaler for mixed precision.

    Args:
        enabled: Whether to enable gradient scaling.

    Returns:
        GradScaler if enabled, else an identity wrapper.
    """
    if enabled and torch.cuda.is_available():
        return GradScaler()
    else:
        return _IdentityScaler()


class _IdentityScaler:
    """Identity scaler that does nothing (for when AMP is disabled)."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Return loss unchanged.

        Args:
            loss: Loss tensor.

        Returns:
            Same loss tensor.
        """
        return loss

    def step(self, optimizer: Any) -> None:
        """No-op step.

        Args:
            optimizer: Optimizer instance.
        """
        pass

    def update(self) -> None:
        """No-op update."""
        pass

    def unscale_(self, optimizer: Any) -> None:
        """No-op unscale.

        Args:
            optimizer: Optimizer instance.
        """
        pass


class MixedPrecisionManager:
    """Context manager for mixed precision training."""

    def __init__(self, enabled: bool = True, dtype: str = "float16") -> None:
        """Initialize mixed precision manager.

        Args:
            enabled: Whether to enable mixed precision.
            dtype: Data type for autocast.
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype
        self.scaler = get_grad_scaler(self.enabled)

    def get_autocast_context(self) -> Any:
        """Get autocast context.

        Returns:
            Autocast context manager.
        """
        return get_autocast_context(self.enabled, self.dtype)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass.

        Args:
            loss: Loss tensor.

        Returns:
            Scaled loss tensor.
        """
        return self.scaler.scale(loss)

    def unscale_gradients(self, optimizer: Any) -> None:
        """Unscale gradients before clipping.

        Args:
            optimizer: Optimizer instance.
        """
        if self.enabled and isinstance(self.scaler, GradScaler):
            self.scaler.unscale_(optimizer)

    def step_scaler(self, optimizer: Any) -> None:
        """Step scaler and optimizer.

        Args:
            optimizer: Optimizer instance.
        """
        self.scaler.step(optimizer)
        self.scaler.update()



