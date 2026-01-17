"""Unified explainer interface for model interpretability.

High-level API that auto-selects appropriate explanation method and
integrates with Phase 1 models and Phase 2/3 workflows.
"""

from typing import Dict, Optional, List, Any, Union
import torch
import torch.nn as nn

from explainability.base import BaseExplainer, get_layer_by_name
from explainability.grad_cam import GradCAM
from explainability.attention_maps import AttentionMapExtractor
from explainability.saliency import VanillaSaliency, SmoothGrad, IntegratedGradients


class LayerGradCAM:
    """Utility class for layer-based Grad-CAM operations."""
    
    @staticmethod
    def suggest_target_layer(model: nn.Module) -> Optional[nn.Module]:
        """Auto-select appropriate target layer for Grad-CAM.
        
        Prioritizes: Conv2d layers from the last conv block or final layers.
        Falls back to last Conv2d layer in the model.
        
        Args:
            model: PyTorch model to analyze.
            
        Returns:
            Suggested nn.Module to use as target layer, or None if not found.
        """
        conv_layers = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)
        
        # Return last conv layer if any exist
        return conv_layers[-1] if conv_layers else None


class Explainer:
    """Unified explainability interface.
    
    Auto-selects and orchestrates explanation methods based on model
    architecture and task type. Provides a single API for all explanation
    techniques.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ) -> None:
        """Initialize unified explainer.

        Args:
            model: Trained model (any Phase 1 architecture).
            device: Device for computation ("cpu" or "cuda").
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()

        # Initialize explanation methods
        self._grad_cam: Optional[GradCAM] = None
        self._vanilla_saliency = VanillaSaliency(model, device)
        self._smooth_grad = SmoothGrad(model, device, num_samples=30)
        self._integrated_gradients = IntegratedGradients(model, device, num_steps=20)
        self._attention_extractor = AttentionMapExtractor(model, device)

    def explain_classification(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        method: str = "grad-cam",
        target_layer: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Explain classification predictions.

        Args:
            inputs: Input images [B, C, H, W].
            targets: Target class indices [B] or None (use predictions).
            method: Explanation method ("grad-cam", "saliency", "smoothgrad", "integrated-gradients", "attention").
            target_layer: For grad-cam, which layer to visualize (auto-selected if None).
            **kwargs: Method-specific arguments.

        Returns:
            Dictionary with explanation artifacts (tensors only, no plots).
        """
        if method == "grad-cam":
            return self._explain_with_grad_cam(
                inputs, targets, "classification", target_layer
            )
        elif method == "saliency":
            return self._vanilla_saliency.explain(
                inputs, targets, task="classification"
            )
        elif method == "smoothgrad":
            return self._smooth_grad.explain(
                inputs, targets, task="classification"
            )
        elif method == "integrated-gradients":
            return self._integrated_gradients.explain(
                inputs, targets, task="classification"
            )
        elif method == "attention":
            return self._attention_extractor.explain(inputs, targets)
        else:
            raise ValueError(f"Unknown method: {method}")

    def explain_segmentation(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        method: str = "grad-cam",
        target_layer: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Explain segmentation predictions.

        Args:
            inputs: Input images [B, C, H, W].
            targets: Target masks [B, H, W] or None.
            method: Explanation method ("grad-cam", "saliency", "smoothgrad", "integrated-gradients").
            target_layer: For grad-cam, which layer to visualize (auto-selected if None).
            **kwargs: Method-specific arguments.

        Returns:
            Dictionary with explanation artifacts.
        """
        if method == "grad-cam":
            return self._explain_with_grad_cam(
                inputs, targets, "segmentation", target_layer
            )
        elif method == "saliency":
            return self._vanilla_saliency.explain(
                inputs, targets, task="segmentation"
            )
        elif method == "smoothgrad":
            return self._smooth_grad.explain(
                inputs, targets, task="segmentation"
            )
        elif method == "integrated-gradients":
            return self._integrated_gradients.explain(
                inputs, targets, task="segmentation"
            )
        elif method == "attention":
            return self._attention_extractor.explain(inputs, targets)
        else:
            raise ValueError(f"Unknown method: {method}")

    def explain_multitask(
        self,
        inputs: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        method: str = "grad-cam",
        task: Optional[str] = None,
        target_layer: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Explain multi-task predictions.

        Args:
            inputs: Input images [B, C, H, W].
            targets: Dictionary with "classification" and/or "segmentation" targets.
            method: Explanation method.
            task: Which task to explain ("classification" or "segmentation"). If None, explain both.
            target_layer: For grad-cam, which layer to visualize.

        Returns:
            Dictionary with explanation artifacts for specified task(s).
        """
        if task is None:
            # Explain both tasks
            results = {}

            if targets and "classification" in targets:
                results["classification"] = self.explain_classification(
                    inputs,
                    targets.get("classification"),
                    method,
                    target_layer,
                )

            if targets and "segmentation" in targets:
                results["segmentation"] = self.explain_segmentation(
                    inputs,
                    targets.get("segmentation"),
                    method,
                    target_layer,
                )

            return results

        elif task == "classification":
            targets_cls = targets.get("classification") if targets else None
            return self.explain_classification(inputs, targets_cls, method, target_layer)

        elif task == "segmentation":
            targets_seg = targets.get("segmentation") if targets else None
            return self.explain_segmentation(inputs, targets_seg, method, target_layer)

        else:
            raise ValueError(f"Unknown task: {task}")

    def explain(
        self,
        inputs: torch.Tensor,
        task: str = "classification",
        targets: Optional[torch.Tensor] = None,
        method: str = "grad-cam",
        target_layer: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Generic explain interface (auto-dispatches by task).

        Args:
            inputs: Input images [B, C, H, W].
            task: Task type ("classification", "segmentation", "multi-task").
            targets: Target labels or masks.
            method: Explanation method.
            target_layer: For grad-cam, which layer to visualize.
            **kwargs: Method-specific arguments.

        Returns:
            Dictionary with explanation artifacts.
        """
        if task == "classification":
            return self.explain_classification(inputs, targets, method, target_layer, **kwargs)
        elif task == "segmentation":
            return self.explain_segmentation(inputs, targets, method, target_layer, **kwargs)
        elif task == "multi-task":
            return self.explain_multitask(inputs, targets, method, target_layer=target_layer)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _explain_with_grad_cam(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor],
        task: str,
        target_layer: Optional[str],
    ) -> Dict[str, torch.Tensor]:
        """Internal method to handle Grad-CAM explanation.

        Args:
            inputs: Input images.
            targets: Target labels or masks.
            task: Task type.
            target_layer: Target layer name or None (auto-select).

        Returns:
            Grad-CAM explanation dictionary.
        """
        # Auto-select target layer if not specified
        if target_layer is None:
            target_layer = LayerGradCAM.suggest_target_layer(self.model)
            if target_layer is None:
                raise ValueError(
                    "Could not auto-select target layer. "
                    "Please specify target_layer explicitly."
                )

        # Create Grad-CAM explainer
        try:
            grad_cam = GradCAM(self.model, target_layer, device=str(self.device))
        except ValueError as e:
            raise ValueError(f"Failed to create Grad-CAM: {e}")

        # Generate explanation
        explanation = grad_cam.explain(inputs, targets, task)

        return explanation

    def list_available_methods(self) -> List[str]:
        """List all available explanation methods.

        Returns:
            List of method names.
        """
        return [
            "grad-cam",
            "saliency",
            "smoothgrad",
            "integrated-gradients",
            "attention",
        ]

    def supported_tasks(self) -> List[str]:
        """List supported task types.

        Returns:
            List of task types.
        """
        return ["classification", "segmentation", "multi-task"]


class ExplainerFactory:
    """Factory for creating task-specific explainer configurations.
    
    Provides pre-configured explainers optimized for different use cases.
    """

    @staticmethod
    def for_classification(
        model: nn.Module,
        device: str = "cpu",
    ) -> Explainer:
        """Create explainer optimized for classification.

        Args:
            model: Classification model.
            device: Device for computation.

        Returns:
            Configured Explainer instance.
        """
        return Explainer(model, device)

    @staticmethod
    def for_segmentation(
        model: nn.Module,
        device: str = "cpu",
    ) -> Explainer:
        """Create explainer optimized for segmentation.

        Args:
            model: Segmentation model.
            device: Device for computation.

        Returns:
            Configured Explainer instance.
        """
        return Explainer(model, device)

    @staticmethod
    def for_multitask(
        model: nn.Module,
        device: str = "cpu",
    ) -> Explainer:
        """Create explainer optimized for multi-task models.

        Args:
            model: Multi-task model.
            device: Device for computation.

        Returns:
            Configured Explainer instance.
        """
        return Explainer(model, device)

    @staticmethod
    def from_checkpoint(
        checkpoint_path: str,
        model_class: type,
        device: str = "cpu",
        **model_kwargs: Any,
    ) -> Explainer:
        """Create explainer from trained checkpoint.

        Args:
            checkpoint_path: Path to saved model checkpoint.
            model_class: Model class to instantiate.
            device: Device for computation.
            **model_kwargs: Arguments for model constructor.

        Returns:
            Configured Explainer with loaded model.
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Instantiate model
        model = model_class(**model_kwargs)

        # Load weights
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        # Create and return explainer
        return Explainer(model, device)
