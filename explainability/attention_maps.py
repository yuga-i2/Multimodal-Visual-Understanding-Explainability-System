"""Attention map extraction for explainability.

Extracts and aggregates attention weights from transformer and hybrid models.
Supports multi-head and multi-layer attention analysis.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn

from explainability.base import (
    BaseExplainer,
    find_modules_by_type,
    normalize_tensor,
)


class AttentionMapExtractor(BaseExplainer):
    """Extract attention weights from transformer models.
    
    Captures and aggregates attention maps from multi-head attention layers.
    Works with standard PyTorch attention modules and custom implementations.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ) -> None:
        """Initialize attention map extractor.

        Args:
            model: Trained model with attention layers.
            device: Device for computation ("cpu" or "cuda").
        """
        super().__init__(model, device)
        self.attention_maps: Dict[str, torch.Tensor] = {}

    def explain(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract attention maps from model.

        Args:
            inputs: Input tensor [B, C, H, W] or [B, N, D] (sequence).
            targets: Unused, for interface compatibility.
            layer_names: Specific attention layers to extract. If None, extract all.

        Returns:
            Dictionary mapping layer names to attention tensors.
            Each tensor has shape [B, N_heads, N_tokens, N_tokens] or similar.
        """
        self.attention_maps.clear()

        # Register hooks on attention layers
        if layer_names is None:
            layer_names = self._find_attention_layers()

        for layer_name in layer_names:
            self._register_attention_hook(layer_name)

        # Forward pass
        with torch.no_grad():
            _ = self.model(inputs.to(self.device))

        # Remove hooks
        self.remove_all_hooks()

        return self.attention_maps.copy()

    def _find_attention_layers(self) -> List[str]:
        """Find all attention layers in model.

        Returns:
            List of layer names containing attention.
        """
        attention_layers = []

        for name, module in self.model.named_modules():
            # Check for standard MultiheadAttention
            if isinstance(module, nn.MultiheadAttention):
                attention_layers.append(name)

            # Check for custom attention patterns
            if hasattr(module, "attention") or hasattr(module, "self_attention"):
                attention_layers.append(name)

            # Check for CBAM or similar channel attention
            if "attention" in name.lower() and not "mha" in name.lower():
                if isinstance(module, nn.Module):
                    attention_layers.append(name)

        return attention_layers

    def _register_attention_hook(self, layer_name: str) -> None:
        """Register hook to capture attention weights.

        Args:
            layer_name: Dot-separated layer name.
        """
        parts = layer_name.split(".")
        current = self.model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return

        # Create hook for this layer
        def attention_hook(module: nn.Module, input: Tuple, output: Union[Tuple, torch.Tensor]) -> None:
            # Try to extract attention weights from output
            if isinstance(output, tuple) and len(output) > 0:
                # Standard attention output: (output, attention_weights)
                if len(output) > 1 and isinstance(output[1], torch.Tensor):
                    self.attention_maps[layer_name] = output[1].detach()
                elif isinstance(output[0], torch.Tensor):
                    self.attention_maps[layer_name] = output[0].detach()
            elif isinstance(output, torch.Tensor):
                self.attention_maps[layer_name] = output.detach()

        # Register hook
        handle = current.register_forward_hook(attention_hook)
        self._hooks[layer_name] = handle

    def supports_task(self, task_type: str) -> bool:
        """Check if method supports task type.

        Args:
            task_type: Task type ("classification", "segmentation", "multi-task").

        Returns:
            Always True (attention can apply to any task).
        """
        return True

    def aggregate_attention(
        self,
        head_dim: int = 0,
        spatial_reduction: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate attention maps across heads.

        Args:
            head_dim: Dimension to aggregate over (usually head dimension).
            spatial_reduction: Whether to average over spatial dimensions.

        Returns:
            Dictionary with aggregated attention maps.
        """
        aggregated = {}

        for layer_name, attn_map in self.attention_maps.items():
            # Average over heads
            if attn_map.ndim >= 3:
                agg_map = attn_map.mean(dim=head_dim)
            else:
                agg_map = attn_map

            # Spatial reduction if requested
            if spatial_reduction and agg_map.ndim > 2:
                agg_map = agg_map.mean(dim=tuple(range(2, agg_map.ndim)))

            aggregated[layer_name] = agg_map

        return aggregated


class AttentionVisualizer:
    """Utilities for visualizing attention maps as heatmaps.
    
    Converts attention tensors to spatial heatmaps for integration
    with input images.
    """

    @staticmethod
    def attention_to_heatmap(
        attention: torch.Tensor,
        input_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Convert attention weights to spatial heatmap.

        Args:
            attention: Attention tensor of various shapes.
            input_shape: Target spatial shape (H, W).

        Returns:
            Heatmap tensor [B, H, W] in [0, 1] range.
        """
        # Flatten and reduce to 2D
        if attention.ndim == 4:
            # [B, N_heads, N, N] -> average over heads -> [B, N, N]
            attention = attention.mean(dim=1)

        if attention.ndim == 3:
            # [B, N, N] -> take last token or average
            # Use last token (typically class token for ViT)
            heatmap = attention[:, -1, :].mean(dim=-1)  # [B]
        elif attention.ndim == 2:
            # [B, N] or [N, N] -> use as-is
            heatmap = attention.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unexpected attention shape: {attention.shape}")

        # Expand to spatial dimensions if needed
        if heatmap.ndim == 1:
            heatmap = heatmap.unsqueeze(-1).unsqueeze(-1)
            heatmap = torch.nn.functional.interpolate(
                heatmap.unsqueeze(0),
                size=input_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        elif heatmap.shape[-2:] != input_shape:
            heatmap = torch.nn.functional.interpolate(
                heatmap.unsqueeze(1),
                size=input_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        # Normalize to [0, 1]
        heatmap = normalize_tensor(heatmap)

        return heatmap

    @staticmethod
    def blend_attention_on_image(
        image: torch.Tensor,
        attention: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Blend attention heatmap with image.

        Args:
            image: Image tensor [C, H, W] in [0, 1] or [0, 255].
            attention: Attention heatmap [H, W] in [0, 1].
            alpha: Blend factor (0=image only, 1=attention only).

        Returns:
            Blended image [3, H, W] in [0, 1].
        """
        # Normalize image
        if image.max() > 1.0:
            image = image / 255.0

        # Convert attention to RGB heatmap (red channel)
        if image.size(0) == 1:
            image_rgb = image.repeat(3, 1, 1)
        elif image.size(0) == 3:
            image_rgb = image
        else:
            image_rgb = image[:3]

        # Create RGB heatmap (red channel only)
        h, w = attention.shape
        heatmap_rgb = torch.zeros(3, h, w, device=attention.device)
        heatmap_rgb[0] = attention  # Red channel = attention

        # Blend
        blended = (1 - alpha) * image_rgb + alpha * heatmap_rgb
        return blended.clamp(0, 1)


class HybridAttentionExtractor(AttentionMapExtractor):
    """Extract attention from hybrid CNN-attention models.
    
    Handles models that combine convolutional and attention mechanisms.
    Automatically detects and extracts from available attention modules.
    """

    def extract_cnn_attention(self) -> Optional[torch.Tensor]:
        """Extract attention from CBAM or similar CNN attention.

        Returns:
            Attention tensor if found, None otherwise.
        """
        try:
            # Look for CBAM modules
            cbam_modules = find_modules_by_type(self.model, type)
            for name, module in self.model.named_modules():
                if "cbam" in name.lower() and hasattr(module, "forward"):
                    return self.attention_maps.get(name)
        except Exception:
            pass

        return None

    def extract_transformer_attention(self) -> Optional[Dict[str, torch.Tensor]]:
        """Extract attention from transformer blocks.

        Returns:
            Dictionary of attention maps if found, None otherwise.
        """
        transformer_attention = {}

        for layer_name, attn_map in self.attention_maps.items():
            if "transformer" in layer_name.lower() or "mha" in layer_name.lower():
                transformer_attention[layer_name] = attn_map

        return transformer_attention if transformer_attention else None
