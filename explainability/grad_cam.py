"""Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

This module provides Grad-CAM for visualizing which regions of input images
influence model predictions. Pure PyTorch implementation using hooks, no
external dependencies beyond torch.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM(nn.Module):
    """Grad-CAM: Gradient-weighted Class Activation Mapping.
    
    Computes class activation maps by weighting feature maps with class-wise gradients.
    Uses PyTorch hooks to capture activations and gradients from target layer.
    
    Reference:
        Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via
        Gradient-based Localization" ICCV 2017.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ) -> None:
        """Initialize Grad-CAM.

        Args:
            model: PyTorch model to explain.
            target_layer: Layer module to visualize (e.g., model.layer4 or model.features[-1]).
                Should be a layer that outputs spatial feature maps (e.g., Conv2d).

        Example:
            >>> model = torchvision.models.resnet50(pretrained=True)
            >>> grad_cam = GradCAM(model, model.layer4[-1])
            >>> cam = grad_cam.generate_cam(image_tensor, class_idx=0)
        """
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        
        # Storage for captured activations and gradients
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        
        # Hook handles for cleanup
        self._hooks = []
        
        # Register hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            """Capture forward activations."""
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            """Capture backward gradients."""
            self.gradients = grad_output[0].detach()

        # Register forward hook
        fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self._hooks.append(fwd_handle)
        
        # Register full backward hook (works with autograd)
        bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks.append(bwd_handle)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def generate_cam(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate Grad-CAM heatmap for input image.

        Args:
            image_tensor: Input image tensor with shape (C, H, W) or (B, C, H, W).
                If 3D, batch dimension is added internally.
            class_idx: Target class index for which to generate CAM. If None,
                uses the class with maximum output score.

        Returns:
            CAM tensor with shape (H, W) normalized to [0, 1].
                For batch input, returns averaged CAM across batch.

        Raises:
            RuntimeError: If gradients or activations were not captured.
        """
        # Handle single image (add batch dimension)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        batch_size = image_tensor.size(0)
        device = image_tensor.device

        # Forward pass with gradient computation
        with torch.enable_grad():
            output = self.model(image_tensor)

            # Determine target class
            if class_idx is None:
                class_idx = output.argmax(dim=1)[0].item()

            # Get target output score
            if output.ndim == 1:
                target_score = output[class_idx]
            else:
                target_score = output[:, class_idx].sum()

            # Backward pass to compute gradients
            self.model.zero_grad()
            target_score.backward(retain_graph=True)

        # Verify hooks captured data
        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Gradients or activations not captured. "
                "Verify target_layer outputs feature maps (e.g., Conv2d output)."
            )

        # Compute weights as mean gradient across spatial dimensions
        # activations shape: (B, C, H, W), gradients shape: (B, C, H, W)
        weights = self.gradients.mean(dim=(2, 3), keepdim=False)  # (B, C)

        # Weight activations by importance: (B, C, H, W) * (B, C, 1, 1)
        weighted_activations = self.activations * weights.unsqueeze(-1).unsqueeze(-1)

        # Sum over channels: (B, C, H, W) -> (B, H, W)
        cam = weighted_activations.sum(dim=1)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Average over batch
        cam = cam.mean(dim=0)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam

    def __del__(self) -> None:
        """Cleanup hooks on deletion."""
        try:
            self._remove_hooks()
        except Exception:
            pass


def overlay_cam_on_image(
    image: torch.Tensor,
    cam: torch.Tensor,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> torch.Tensor:
    """Overlay Grad-CAM heatmap on original image.

    Helper function for visualizing Grad-CAM results. Creates a blended
    visualization of the input image and attention heatmap.

    Args:
        image: Original image tensor with shape (C, H, W). 
            Values should be in range [0, 1] or [0, 255].
        cam: CAM heatmap with shape (H, W) in range [0, 1].
        alpha: Blend factor for overlay (0=image only, 1=CAM only).
            Default 0.5 for balanced visualization.
        colormap: Colormap name. Currently supports "jet" (default),
            "hot", "cool", "gray".

    Returns:
        Overlaid image tensor with shape (C, H, W) in range [0, 1].

    Example:
        >>> image = torch.rand(3, 224, 224)
        >>> cam = torch.rand(224, 224)
        >>> overlay = overlay_cam_on_image(image, cam, alpha=0.5)
    """
    # Ensure inputs are on CPU and correct shape
    if image.ndim != 3:
        raise ValueError(f"Image must have shape (C, H, W), got {image.shape}")
    if cam.ndim != 2:
        raise ValueError(f"CAM must have shape (H, W), got {cam.shape}")

    image = image.detach().cpu()
    cam = cam.detach().cpu()

    # Resize CAM to match image if needed
    if cam.shape != image.shape[1:]:
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=image.shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    # Normalize image to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0

    # Apply colormap to CAM
    if colormap == "jet":
        cam_rgb = _jet_colormap(cam)
    elif colormap == "hot":
        cam_rgb = _hot_colormap(cam)
    elif colormap == "cool":
        cam_rgb = _cool_colormap(cam)
    elif colormap == "gray":
        cam_rgb = cam.unsqueeze(0).repeat(3, 1, 1)
    else:
        raise ValueError(f"Unknown colormap: {colormap}. Use 'jet', 'hot', 'cool', or 'gray'.")

    # Ensure image is RGB
    if image.size(0) == 1:
        image = image.repeat(3, 1, 1)
    elif image.size(0) == 4:
        image = image[:3]

    # Blend: (1 - alpha) * image + alpha * cam_rgb
    overlay = (1 - alpha) * image + alpha * cam_rgb
    return overlay.clamp(0, 1)


def _jet_colormap(x: torch.Tensor) -> torch.Tensor:
    """Convert grayscale tensor [0, 1] to jet colormap RGB.

    Jet colormap: blue (low) -> cyan -> green -> yellow -> red (high).

    Args:
        x: Grayscale tensor with values in [0, 1].

    Returns:
        RGB tensor with shape (3, H, W) and values in [0, 1].
    """
    x = x.clamp(0, 1)

    # Red channel: low in first half, high in second half
    r = torch.where(x < 0.35, torch.zeros_like(x), 
        torch.where(x < 0.65, (x - 0.35) / 0.3, 1.0))

    # Green channel: rises then falls
    g = torch.where(x < 0.125, torch.zeros_like(x),
        torch.where(x < 0.375, (x - 0.125) / 0.25,
        torch.where(x < 0.625, 1.0,
        torch.where(x < 0.875, (0.875 - x) / 0.25, torch.zeros_like(x)))))

    # Blue channel: high initially, low at end
    b = torch.where(x < 0.65, 1.0,
        torch.where(x < 0.875, (0.875 - x) / 0.25, torch.zeros_like(x)))

    return torch.stack([r, g, b], dim=0)


def _hot_colormap(x: torch.Tensor) -> torch.Tensor:
    """Convert grayscale tensor [0, 1] to hot colormap RGB.

    Hot colormap: black -> red -> yellow -> white.

    Args:
        x: Grayscale tensor with values in [0, 1].

    Returns:
        RGB tensor with shape (3, H, W) and values in [0, 1].
    """
    x = x.clamp(0, 1)
    
    # Red channel: rises early
    r = torch.where(x < 0.33, x / 0.33, 1.0)
    
    # Green channel: rises later
    g = torch.where(x < 0.33, torch.zeros_like(x),
        torch.where(x < 0.66, (x - 0.33) / 0.33, 1.0))
    
    # Blue channel: rises last
    b = torch.where(x < 0.66, torch.zeros_like(x),
        torch.where(x < 1.0, (x - 0.66) / 0.34, 1.0))
    
    return torch.stack([r, g, b], dim=0)


def _cool_colormap(x: torch.Tensor) -> torch.Tensor:
    """Convert grayscale tensor [0, 1] to cool colormap RGB.

    Cool colormap: cyan -> magenta.

    Args:
        x: Grayscale tensor with values in [0, 1].

    Returns:
        RGB tensor with shape (3, H, W) and values in [0, 1].
    """
    x = x.clamp(0, 1)
    
    # Red channel: rises from 0 to 1
    r = x
    
    # Green channel: decreases from 1 to 0
    g = 1.0 - x
    
    # Blue channel: constant 1
    b = torch.ones_like(x)
    
    return torch.stack([r, g, b], dim=0)
