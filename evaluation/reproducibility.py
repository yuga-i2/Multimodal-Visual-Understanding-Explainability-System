"""Reproducibility utilities for deterministic training and evaluation.

Provides centralized seed control and reproducibility configuration
for Python, NumPy, and PyTorch (CPU and CUDA).
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU)
    - PyTorch (CUDA, if available)

    Args:
        seed: Random seed value.

    Example:
        >>> from evaluation.reproducibility import set_seed
        >>> set_seed(42)
        >>> # All subsequent random operations will be deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_cudnn_determinism(enabled: bool = True) -> None:
    """Enable or disable cuDNN determinism.

    When enabled, cuDNN will use deterministic algorithms.
    This may reduce performance but ensures reproducibility.

    Warning:
        Some operations do not have deterministic implementations
        and will raise an error if called with determinism enabled.

    Args:
        enabled: Whether to enable cuDNN determinism.

    Example:
        >>> from evaluation.reproducibility import enable_cudnn_determinism
        >>> enable_cudnn_determinism(True)
        >>> # Training will now be deterministic on GPU
    """
    if enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
