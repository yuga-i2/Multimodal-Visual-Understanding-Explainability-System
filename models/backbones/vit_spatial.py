"""Vision Transformer (ViT) backbone for feature extraction.

This implementation focuses on spatial feature extraction suitable for downstream
segmentation and classification tasks. Unlike standard ViT, this returns spatial
feature maps rather than flattened sequences.
"""

from typing import List, Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings with spatial awareness."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        """Initialize patch embedding.

        Args:
            image_size: Input image size (assumed square).
            patch_size: Patch size (assumed square).
            in_channels: Input channels (e.g., 3 for RGB).
            embed_dim: Embedding dimension.
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.grid_h = self.grid_w = image_size // patch_size

        # Conv projection: treats patches as conv operation
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Patch embeddings [B, embed_dim, grid_h, grid_w] (maintains spatial dims).
        """
        return self.proj(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, embed_dim: int, num_heads: int = 8, attn_drop: float = 0.0) -> None:
        """Initialize attention.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            attn_drop: Attention dropout rate.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention forward.

        Args:
            x: Input tensor [B, N, embed_dim] where N = num_patches.

        Returns:
            Output tensor [B, N, embed_dim].
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Merge heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLPBlock(nn.Module):
    """Feed-forward block."""

    def __init__(self, dim: int, mlp_dim: int = 2048, dropout: float = 0.0) -> None:
        """Initialize MLP.

        Args:
            dim: Input/output dimension.
            mlp_dim: Hidden dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block: attention + MLP with residuals."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        """Initialize transformer block.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            mlp_dim: Hidden dimension in MLP.
            dropout: Dropout rate.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads=num_heads, attn_drop=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_dim=mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input tensor [B, N, embed_dim].

        Returns:
            Output tensor [B, N, embed_dim].
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViTBackbone(nn.Module):
    """Vision Transformer backbone with spatial feature output.

    Unlike standard ViT that outputs class tokens or flattened sequences,
    this variant maintains spatial structure for segmentation tasks.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
    ) -> None:
        """Initialize ViT backbone.

        Args:
            image_size: Input image size (square).
            patch_size: Patch size (square).
            in_channels: Input channels.
            embed_dim: Embedding dimension.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_dim: Hidden dimension in MLP.
            dropout: Dropout rate.
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_h = self.grid_w = image_size // patch_size

        # Patch embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)

        # Class token (for classification, optional)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings: (1 + num_patches, embed_dim)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )

        self.ln = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Transformer features [B, num_patches + 1, embed_dim].
        """
        B = x.shape[0]

        # Patch embedding: [B, embed_dim, grid_h, grid_w]
        x = self.patch_embed(x)

        # Flatten to sequence: [B, embed_dim, grid_h*grid_w] -> [B, grid_h*grid_w, embed_dim]
        x = x.flatten(2).transpose(1, 2)

        # Add class token: [B, 1 + num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Layer norm
        x = self.ln(x)

        return x

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning spatial feature maps.

        Reshapes sequence output back to spatial grid for downstream tasks.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Spatial features [B, embed_dim, grid_h, grid_w].
        """
        seq_features = self.forward(x)

        # Remove class token: [B, 1 + num_patches, embed_dim] -> [B, num_patches, embed_dim]
        patch_features = seq_features[:, 1:, :]

        # Reshape to spatial: [B, num_patches, embed_dim] -> [B, embed_dim, grid_h, grid_w]
        B = patch_features.shape[0]
        spatial = patch_features.transpose(1, 2).reshape(B, self.embed_dim, self.grid_h, self.grid_w)

        return spatial

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only spatial features (no class token).

        Convenience method for models that only need patch embeddings.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Spatial features [B, embed_dim, grid_h, grid_w].
        """
        return self.forward_spatial(x)


def create_vit_backbone(
    image_size: int = 224,
    patch_size: int = 16,
    in_channels: int = 3,
    preset: Literal["tiny", "small", "base", "large"] = "base",
) -> ViTBackbone:
    """Factory function to create ViT backbone with preset configurations.

    Args:
        image_size: Input image size.
        patch_size: Patch size.
        in_channels: Input channels.
        preset: Model size preset.
                - tiny: embed_dim=192, depth=12, heads=3
                - small: embed_dim=384, depth=12, heads=6
                - base: embed_dim=768, depth=12, heads=12
                - large: embed_dim=1024, depth=24, heads=16

    Returns:
        Initialized ViTBackbone.
    """
    configs = {
        "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3, "mlp_dim": 768},
        "small": {"embed_dim": 384, "depth": 12, "num_heads": 6, "mlp_dim": 1536},
        "base": {"embed_dim": 768, "depth": 12, "num_heads": 12, "mlp_dim": 3072},
        "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "mlp_dim": 4096},
    }

    config = configs[preset]
    return ViTBackbone(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        **config,
    )
