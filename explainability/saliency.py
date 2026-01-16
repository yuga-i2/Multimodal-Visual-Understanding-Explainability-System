"""Saliency map methods for model explainability.

Implements gradient-based saliency methods:
- Vanilla saliency (input gradients)
- SmoothGrad (noise-averaged gradients)
"""

from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from explainability.base import (
    BaseExplainer,
    GradientContext,
    enable_gradient,
    normalize_tensor,
)


class VanillaSaliency(BaseExplainer):
    """Vanilla saliency maps via input gradient visualization.
    
    Computes gradients of model output with respect to input, showing
    which input pixels most influence the prediction.
    
    Reference:
        Simonyan et al. "Deep Inside Convolutional Networks:
        Visualising Image Classification Models and Saliency Maps" 2014.
    """

    def explain(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        task: str = "classification",
    ) -> Dict[str, torch.Tensor]:
        """Generate saliency map.

        Args:
            inputs: Input tensor [B, C, H, W].
            targets: Target class indices [B] for classification or
                     target masks [B, H, W] for segmentation.
            task: Task type ("classification" or "segmentation").

        Returns:
            Dictionary with:
            - "saliency": Saliency map [B, C, H, W]
            - "magnitude": Magnitude of gradient [B, 1, H, W]
        """
        inputs = inputs.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)

        # Enable gradients for input
        inputs = enable_gradient(inputs)

        with GradientContext(self.model):
            # Forward pass
            outputs = self.model(inputs)

            # Compute loss for backward
            if task == "classification":
                loss = self._classification_loss(outputs, targets)
            elif task == "segmentation":
                loss = self._segmentation_loss(outputs, targets)
            else:
                raise ValueError(f"Unknown task: {task}")

            # Backward pass
            self.model.zero_grad()
            loss.backward(retain_graph=False)

        # Extract gradients
        gradients = inputs.grad
        if gradients is None:
            raise RuntimeError("Gradients not computed")

        # Compute saliency (absolute value of gradients)
        saliency = torch.abs(gradients).detach()

        # Compute magnitude (L2 norm across channel dimension)
        magnitude = torch.norm(gradients.detach(), dim=1, keepdim=True)
        magnitude = normalize_tensor(magnitude)

        return {
            "saliency": saliency,
            "magnitude": magnitude,
        }

    def _classification_loss(
        self,
        outputs: torch.Tensor,
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss for classification.

        Args:
            outputs: Model logits [B, C].
            targets: Target class indices [B] or None.

        Returns:
            Scalar loss tensor.
        """
        if targets is None:
            # Use predicted class
            targets = outputs.argmax(dim=1)

        # Create one-hot targets
        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Compute weighted sum (higher for target class)
        loss = (outputs * one_hot).sum() / outputs.size(0)
        return loss

    def _segmentation_loss(
        self,
        outputs: torch.Tensor,
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss for segmentation.

        Args:
            outputs: Model logits [B, C, H, W].
            targets: Target masks [B, H, W] or None.

        Returns:
            Scalar loss tensor.
        """
        if isinstance(outputs, dict):
            outputs = outputs.get("segmentation", outputs[list(outputs.keys())[0]])

        if targets is not None:
            loss = F.cross_entropy(outputs, targets, reduction="mean")
        else:
            # Use mean activation as proxy
            loss = outputs.mean()

        return loss

    def supports_task(self, task_type: str) -> bool:
        """Check if method supports task type.

        Args:
            task_type: Task type ("classification", "segmentation", "multi-task").

        Returns:
            True for supported types.
        """
        return task_type in ("classification", "segmentation", "multi-task")


class SmoothGrad(BaseExplainer):
    """SmoothGrad: gradient-based saliency with noise averaging.
    
    Computes saliency maps by averaging gradients over multiple noisy
    versions of the input. Reduces noise artifacts in vanilla gradients.
    
    Reference:
        Smilkov et al. "SmoothGrad: removing noise by adding noise"
        arXiv:1706.03762, 2017.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        num_samples: int = 50,
        noise_level: float = 0.15,
    ) -> None:
        """Initialize SmoothGrad.

        Args:
            model: Trained model.
            device: Device for computation.
            num_samples: Number of noise samples to average over.
            noise_level: Standard deviation of Gaussian noise (fraction of input range).
        """
        super().__init__(model, device)
        self.num_samples = num_samples
        self.noise_level = noise_level

    def explain(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        task: str = "classification",
    ) -> Dict[str, torch.Tensor]:
        """Generate smoothed saliency map.

        Args:
            inputs: Input tensor [B, C, H, W].
            targets: Target class indices [B] or masks [B, H, W].
            task: Task type ("classification" or "segmentation").

        Returns:
            Dictionary with:
            - "saliency": Smoothed saliency map [B, C, H, W]
            - "magnitude": Magnitude of smoothed gradients [B, 1, H, W]
        """
        inputs = inputs.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)

        # Accumulate gradients
        accumulated_gradients = torch.zeros_like(inputs)
        accumulated_magnitude = torch.zeros(
            inputs.shape[0], 1, inputs.shape[2], inputs.shape[3],
            device=self.device, dtype=inputs.dtype
        )

        for _ in range(self.num_samples):
            # Add noise to input
            noise = torch.randn_like(inputs) * self.noise_level
            noisy_input = inputs + noise
            noisy_input = enable_gradient(noisy_input)

            with GradientContext(self.model):
                # Forward pass
                outputs = self.model(noisy_input)

                # Compute loss
                if task == "classification":
                    loss = self._classification_loss(outputs, targets)
                elif task == "segmentation":
                    loss = self._segmentation_loss(outputs, targets)
                else:
                    raise ValueError(f"Unknown task: {task}")

                # Backward pass
                self.model.zero_grad()
                loss.backward(retain_graph=False)

            # Accumulate gradients
            gradients = noisy_input.grad
            if gradients is not None:
                accumulated_gradients += torch.abs(gradients.detach())
                magnitude = torch.norm(gradients.detach(), dim=1, keepdim=True)
                accumulated_magnitude += magnitude

        # Average over samples
        smoothed_saliency = accumulated_gradients / self.num_samples
        smoothed_magnitude = accumulated_magnitude / self.num_samples

        # Normalize magnitude to [0, 1]
        smoothed_magnitude = normalize_tensor(smoothed_magnitude)

        return {
            "saliency": smoothed_saliency,
            "magnitude": smoothed_magnitude,
        }

    def _classification_loss(
        self,
        outputs: torch.Tensor,
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss for classification."""
        if targets is None:
            targets = outputs.argmax(dim=1)

        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        loss = (outputs * one_hot).sum() / outputs.size(0)
        return loss

    def _segmentation_loss(
        self,
        outputs: torch.Tensor,
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss for segmentation."""
        if isinstance(outputs, dict):
            outputs = outputs.get("segmentation", outputs[list(outputs.keys())[0]])

        if targets is not None:
            loss = F.cross_entropy(outputs, targets, reduction="mean")
        else:
            loss = outputs.mean()

        return loss

    def supports_task(self, task_type: str) -> bool:
        """Check if method supports task type.

        Args:
            task_type: Task type ("classification", "segmentation", "multi-task").

        Returns:
            True for supported types.
        """
        return task_type in ("classification", "segmentation", "multi-task")


class IntegratedGradients(BaseExplainer):
    """Integrated Gradients for attributing predictions to input features.
    
    Computes path integral of gradients from baseline to input, providing
    a theoretically grounded attribution method.
    
    Reference:
        Sundararajan et al. "Axiomatic Attribution for Deep Networks"
        ICML 2017.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        num_steps: int = 50,
    ) -> None:
        """Initialize Integrated Gradients.

        Args:
            model: Trained model.
            device: Device for computation.
            num_steps: Number of steps for path integral approximation.
        """
        super().__init__(model, device)
        self.num_steps = num_steps

    def explain(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        baseline: Optional[torch.Tensor] = None,
        task: str = "classification",
    ) -> Dict[str, torch.Tensor]:
        """Compute integrated gradients.

        Args:
            inputs: Input tensor [B, C, H, W].
            targets: Target class indices [B] or masks [B, H, W].
            baseline: Baseline input (default: black image).
            task: Task type ("classification" or "segmentation").

        Returns:
            Dictionary with:
            - "attributions": Attribution maps [B, C, H, W]
            - "accumulated": Integrated gradients [B, C, H, W]
        """
        inputs = inputs.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)

        # Use black image as default baseline
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        else:
            baseline = baseline.to(self.device)

        # Accumulate gradients
        accumulated = torch.zeros_like(inputs)

        for step in range(self.num_steps):
            # Interpolate between baseline and input
            alpha = step / self.num_steps
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated = enable_gradient(interpolated)

            with GradientContext(self.model):
                outputs = self.model(interpolated)

                if task == "classification":
                    loss = self._classification_loss(outputs, targets)
                elif task == "segmentation":
                    loss = self._segmentation_loss(outputs, targets)
                else:
                    raise ValueError(f"Unknown task: {task}")

                self.model.zero_grad()
                loss.backward(retain_graph=False)

            # Accumulate gradients
            gradients = interpolated.grad
            if gradients is not None:
                accumulated += gradients.detach()

        # Average over steps and multiply by (input - baseline)
        accumulated = accumulated / self.num_steps
        attributions = accumulated * (inputs - baseline)

        # Normalize to [0, 1]
        attributions = normalize_tensor(torch.abs(attributions))

        return {
            "attributions": attributions,
            "accumulated": accumulated,
        }

    def _classification_loss(
        self,
        outputs: torch.Tensor,
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss for classification."""
        if targets is None:
            targets = outputs.argmax(dim=1)

        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        loss = (outputs * one_hot).sum() / outputs.size(0)
        return loss

    def _segmentation_loss(
        self,
        outputs: torch.Tensor,
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss for segmentation."""
        if isinstance(outputs, dict):
            outputs = outputs.get("segmentation", outputs[list(outputs.keys())[0]])

        if targets is not None:
            loss = F.cross_entropy(outputs, targets, reduction="mean")
        else:
            loss = outputs.mean()

        return loss

    def supports_task(self, task_type: str) -> bool:
        """Check if method supports task type."""
        return task_type in ("classification", "segmentation", "multi-task")
