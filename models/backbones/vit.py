"""Vision Transformer (ViT): An Image is Worth 16x16 Words.

Reference: Dosovitskiy et al., 2020 - https://arxiv.org/abs/2010.11929
"""

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        """Initialize patch embedding.

        Args:
            image_size: Size of input image (assumed square).
            patch_size: Size of each patch (assumed square).
            in_channels: Number of input channels (e.g., 3 for RGB).
            embed_dim: Embedding dimension.
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        # Verify image dimensions are divisible by patch size
        assert (
            image_size % patch_size == 0
        ), f"Image size ({image_size}) must be divisible by patch size ({patch_size})"

        # Project patches to embedding dimension
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings.

        Args:
            x: Input image tensor of shape (batch, channels, height, width).

        Returns:
            Patch embeddings of shape (batch, num_patches, embed_dim).
        """
        batch_size = x.shape[0]

        # Reshape to patches: (batch, channels, height, width) ->
        # (batch, num_patches, patch_dim)
        x = x.reshape(
            batch_size,
            x.shape[1],
            self.image_size // self.patch_size,
            self.patch_size,
            self.image_size // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(batch_size, self.num_patches, self.patch_dim)

        # Project to embedding dimension
        x = self.proj(x)

        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
    ) -> None:
        """Initialize multi-head attention.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            attn_dropout: Dropout probability for attention weights.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        batch_size, seq_len, embed_dim = x.shape

        # Layer normalization
        x = self.norm(x)

        # Project to Q, K, V
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Combine heads
        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, embed_dim)

        # Output projection
        output = self.proj(context)

        return output


class MLPBlock(nn.Module):
    """Feed-forward MLP block."""

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize MLP block.

        Args:
            embed_dim: Embedding dimension.
            mlp_dim: Hidden dimension in MLP.
            dropout: Dropout probability.
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP block.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        return self.mlp(self.norm(x))


class TransformerBlock(nn.Module):
    """Transformer block: attention + MLP with residual connections."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ) -> None:
        """Initialize transformer block.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            mlp_dim: Hidden dimension in MLP.
            attn_dropout: Dropout in attention.
            mlp_dropout: Dropout in MLP.
        """
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, attn_dropout)
        self.mlp = MLPBlock(embed_dim, mlp_dim, mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) for image classification.

    Features:
    - Patch-based image tokenization
    - Multi-head self-attention
    - Transformer encoder stack
    - Optional classification head
    - Configurable architecture
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        pool: Literal["cls", "mean"] = "cls",
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        """Initialize Vision Transformer.

        Args:
            image_size: Input image size (assumed square).
            patch_size: Patch size (assumed square).
            in_channels: Number of input channels (3 for RGB).
            num_classes: Number of output classes.
            embed_dim: Embedding/hidden dimension.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_dim: Hidden dimension in MLP blocks.
            pool: Pooling strategy ('cls' or 'mean').
            attn_dropout: Dropout in attention.
            proj_dropout: Dropout in projections.
            emb_dropout: Dropout after embeddings.
        """
        super().__init__()

        assert pool in {
            "cls",
            "mean",
        }, "pool must be 'cls' or 'mean'"
        assert (
            embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.pool = pool

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embedding.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        if pool == "cls":
            self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.emb_dropout = nn.Dropout(emb_dropout)

        # Transformer encoder
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    attn_dropout=attn_dropout,
                    mlp_dropout=proj_dropout,
                )
                for _ in range(depth)
            ]
        )

        # Output
        self.norm = nn.LayerNorm(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Vision Transformer.

        Args:
            x: Input image tensor of shape (batch, channels, height, width).

        Returns:
            Classification logits of shape (batch, num_classes).
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embedding(x)  # (batch, num_patches, embed_dim)

        # Add CLS token if using CLS pooling
        if self.pool == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches + 1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embedding[: x.shape[1]]
        x = self.emb_dropout(x)

        # Transformer blocks
        for block in self.transformer:
            x = block(x)

        # Layer normalization
        x = self.norm(x)

        # Pooling
        if self.pool == "cls":
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        # Classification head
        x = self.head(x)

        return x

    def get_num_patches(self) -> int:
        """Get number of patches.

        Returns:
            Number of patches in the image.
        """
        return self.patch_embedding.num_patches


def create_vit(
    image_size: int = 224,
    patch_size: int = 16,
    in_channels: int = 3,
    num_classes: int = 1000,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_dim: int = 3072,
    pool: Literal["cls", "mean"] = "cls",
    attn_dropout: float = 0.0,
    proj_dropout: float = 0.0,
    emb_dropout: float = 0.0,
    pretrained_path: Optional[str] = None,
) -> VisionTransformer:
    """Factory function to create a Vision Transformer model.

    Args:
        image_size: Input image size.
        patch_size: Patch size.
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_dim: MLP hidden dimension.
        pool: Pooling strategy.
        attn_dropout: Attention dropout.
        proj_dropout: Projection dropout.
        emb_dropout: Embedding dropout.
        pretrained_path: Optional path to pretrained weights.

    Returns:
        Initialized VisionTransformer model.
    """
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        pool=pool,
        attn_dropout=attn_dropout,
        proj_dropout=proj_dropout,
        emb_dropout=emb_dropout,
    )

    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    return model


# Common ViT configurations
def vit_tiny(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Tiny (12M parameters)."""
    return VisionTransformer(
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_dim=768,
        num_classes=num_classes,
        **kwargs,
    )


def vit_small(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Small (22M parameters)."""
    return VisionTransformer(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_dim=1536,
        num_classes=num_classes,
        **kwargs,
    )


def vit_base(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Base (86M parameters)."""
    return VisionTransformer(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        num_classes=num_classes,
        **kwargs,
    )


def vit_large(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Large (304M parameters)."""
    return VisionTransformer(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_dim=4096,
        num_classes=num_classes,
        **kwargs,
    )
