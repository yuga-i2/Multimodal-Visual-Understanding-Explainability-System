"""Base abstractions and utilities for explainability methods.

Provides common functionality for all explainability techniques:
- Hook registration for feature extraction
- Safe gradient computation
- Device management
- Type checking and validation
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn


class BaseExplainer(ABC):
    """Abstract base class for all explainability methods.
    
    Defines common interface and utilities for gradient-based, attention-based,
    and activation-based explanation methods.
    """

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        """Initialize explainer with model.

        Args:
            model: Trained neural network (any nn.Module).
            device: Device for computation ("cpu" or "cuda").
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()

        # Hook storage for feature extraction
        self._hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._activations: Dict[str, torch.Tensor] = {}

    @abstractmethod
    def explain(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Generate explanation for model predictions.

        Args:
            inputs: Input tensor [B, C, H, W].
            targets: Optional target labels or masks.
            **kwargs: Method-specific arguments.

        Returns:
            Dictionary with explanation artifacts (tensors).
        """
        pass

    @abstractmethod
    def supports_task(self, task_type: str) -> bool:
        """Check if explainer supports given task.

        Args:
            task_type: Task type ("classification", "segmentation", "multi-task").

        Returns:
            True if method can explain this task type.
        """
        pass

    def register_hook(
        self,
        module: nn.Module,
        hook_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], None],
        hook_name: str,
    ) -> None:
        """Register forward hook on module to capture activations.

        Args:
            module: Module to hook.
            hook_fn: Hook function taking (module, input, output).
            hook_name: Name for storing hook handle.
        """
        handle = module.register_forward_hook(hook_fn)
        self._hooks[hook_name] = handle

    def remove_hook(self, hook_name: str) -> None:
        """Remove hook by name.

        Args:
            hook_name: Name of hook to remove.
        """
        if hook_name in self._hooks:
            self._hooks[hook_name].remove()
            del self._hooks[hook_name]

    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook_name in list(self._hooks.keys()):
            self.remove_hook(hook_name)

    def activation_hook(self, name: str) -> Callable:
        """Create hook function that stores activations.

        Args:
            name: Name to store activation under.

        Returns:
            Hook function for register_forward_hook.
        """
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            self._activations[name] = output.detach()

        return hook

    def get_activation(self, name: str) -> Optional[torch.Tensor]:
        """Retrieve stored activation.

        Args:
            name: Name of activation.

        Returns:
            Activation tensor or None if not found.
        """
        return self._activations.get(name)

    def clear_activations(self) -> None:
        """Clear all stored activations."""
        self._activations.clear()

    def __del__(self) -> None:
        """Cleanup hooks on deletion."""
        self.remove_all_hooks()


class GradientContext:
    """Context manager for safe gradient computation.
    
    Ensures gradients are enabled during computation and properly
    cleans up afterward.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize context.

        Args:
            model: Model that may have gradients disabled.
        """
        self.model = model
        self.was_training = model.training
        self.had_requires_grad = None

    def __enter__(self) -> None:
        """Enable gradients for computation."""
        self.model.train()
        # Collect which parameters require grad
        self.had_requires_grad = {}
        for name, param in self.model.named_parameters():
            self.had_requires_grad[name] = param.requires_grad
            param.requires_grad = True

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Restore original gradient state."""
        # Restore original requires_grad
        for name, param in self.model.named_parameters():
            if name in self.had_requires_grad:
                param.requires_grad = self.had_requires_grad[name]

        # Restore training mode
        if self.was_training:
            self.model.train()
        else:
            self.model.eval()


def enable_gradient(inputs: torch.Tensor) -> torch.Tensor:
    """Ensure tensor requires gradients for explanation computation.

    Args:
        inputs: Input tensor.

    Returns:
        Input tensor with requires_grad=True.
    """
    if not inputs.requires_grad:
        inputs = inputs.clone().detach().requires_grad_(True)
    return inputs


def safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safe division with epsilon to avoid division by zero.

    Args:
        a: Numerator tensor.
        b: Denominator tensor.
        eps: Small constant to add to denominator.

    Returns:
        Result of a / (b + eps).
    """
    return a / (b + eps)


def normalize_tensor(
    tensor: torch.Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize tensor to [0, 1] range.

    Args:
        tensor: Tensor to normalize.
        dim: Dimension(s) to compute min/max over. If None, global normalization.
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor in [0, 1].
    """
    if dim is None:
        # Global normalization
        min_val = tensor.min()
        max_val = tensor.max()
    else:
        # Per-element normalization along specified dimension
        min_val = tensor.amin(dim=dim, keepdim=True)
        max_val = tensor.amax(dim=dim, keepdim=True)

    normalized = safe_div(tensor - min_val, max_val - min_val, eps)
    return torch.clamp(normalized, 0, 1)


def spatial_average(tensor: torch.Tensor) -> torch.Tensor:
    """Average tensor over spatial dimensions (H, W).

    Args:
        tensor: Input tensor [B, C, H, W] or [B, C, H, W, ...].

    Returns:
        Averaged tensor [B, C].
    """
    # Average over all dimensions except batch and channel
    return tensor.mean(dim=tuple(range(2, tensor.ndim)))


def get_layer_by_name(
    model: nn.Module,
    layer_name: str,
) -> Optional[nn.Module]:
    """Get layer from model by name (dot-separated path).

    Args:
        model: Root module.
        layer_name: Dot-separated layer path (e.g., "layer1.0.conv1").

    Returns:
        Module if found, None otherwise.
    """
    parts = layer_name.split(".")
    current = model
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current


def list_named_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """List all named modules in model.

    Args:
        model: Root module.

    Returns:
        List of (name, module) tuples.
    """
    return list(model.named_modules())


def find_modules_by_type(
    model: nn.Module,
    module_type: type,
) -> List[Tuple[str, nn.Module]]:
    """Find all modules of given type in model.

    Args:
        model: Root module.
        module_type: Type to search for (e.g., nn.Conv2d).

    Returns:
        List of (name, module) tuples matching type.
    """
    result = []
    for name, module in model.named_modules():
        if isinstance(module, module_type):
            result.append((name, module))
    return result
