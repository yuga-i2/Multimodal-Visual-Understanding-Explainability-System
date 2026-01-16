"""Phase 4 Integration Examples.

Demonstrates usage of explainability methods with Phase 1 models
and Phase 2/3 workflows.
"""

import torch
import torch.nn as nn
from typing import Tuple


# ============================================================================
# Example 1: Grad-CAM for Classification
# ============================================================================


def example_grad_cam_classification() -> None:
    """Demonstrate Grad-CAM for classification."""
    print("\n" + "=" * 70)
    print("Example 1: Grad-CAM for Classification")
    print("=" * 70)

    from explainability.grad_cam import GradCAM, LayerGradCAM

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dummy classification model (CNN)
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 10),
    ).to(device)

    # Find Conv2d layers
    conv_layers = LayerGradCAM.find_conv_layers(model)
    print(f"Found {len(conv_layers)} Conv2d layers: {list(conv_layers.keys())}")

    # Auto-suggest target layer
    target_layer = LayerGradCAM.suggest_target_layer(model)
    print(f"Suggested target layer: {target_layer}")

    # Create Grad-CAM
    try:
        grad_cam = GradCAM(model, target_layer, device=device)

        # Generate CAM
        images = torch.randn(4, 3, 224, 224).to(device)
        class_labels = torch.tensor([0, 1, 2, 3]).to(device)

        results = grad_cam.explain(images, class_labels, task="classification")

        print(f"\nGrad-CAM Results:")
        print(f"  grad_cam shape: {results['grad_cam'].shape}")
        print(f"  grad_cam range: [{results['grad_cam'].min():.4f}, {results['grad_cam'].max():.4f}]")
        print(f"  activations shape: {results['activations'].shape}")
        print(f"  gradients shape: {results['gradients'].shape}")

        # Analyze heatmap
        heatmap = results["grad_cam"][0, 0]  # First batch, single channel
        top_val = heatmap.max().item()
        print(f"\n  First sample - Max activation: {top_val:.4f}")
        print(f"  Regions with high activation: {(heatmap > 0.7).sum().item()} pixels")

    except Exception as e:
        print(f"  Error: {e}")


# ============================================================================
# Example 2: Vanilla Saliency for Segmentation
# ============================================================================


def example_vanilla_saliency() -> None:
    """Demonstrate vanilla saliency for segmentation."""
    print("\n" + "=" * 70)
    print("Example 2: Vanilla Saliency Maps")
    print("=" * 70)

    from explainability.saliency import VanillaSaliency

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dummy segmentation model
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 21, kernel_size=1),  # 21 classes
    ).to(device)

    explainer = VanillaSaliency(model, device=device)

    # Generate saliency
    images = torch.randn(2, 3, 256, 256).to(device)
    masks = torch.randint(0, 21, (2, 256, 256)).to(device)

    results = explainer.explain(images, masks, task="segmentation")

    print(f"Saliency Results:")
    print(f"  saliency shape: {results['saliency'].shape}")
    print(f"  magnitude shape: {results['magnitude'].shape}")
    print(f"  magnitude range: [{results['magnitude'].min():.4f}, {results['magnitude'].max():.4f}]")

    # Analyze saliency
    magnitude = results["magnitude"][0, 0]  # [H, W]
    high_sensitivity_pixels = (magnitude > 0.5).sum().item()
    print(f"\n  Pixels with high sensitivity: {high_sensitivity_pixels}")
    print(f"  Total pixels: {magnitude.numel()}")
    print(f"  Percentage: {100 * high_sensitivity_pixels / magnitude.numel():.2f}%")


# ============================================================================
# Example 3: SmoothGrad
# ============================================================================


def example_smooth_grad() -> None:
    """Demonstrate SmoothGrad."""
    print("\n" + "=" * 70)
    print("Example 3: SmoothGrad (Noise-Averaged Saliency)")
    print("=" * 70)

    from explainability.saliency import SmoothGrad

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = nn.Sequential(
        nn.Linear(224 * 224 * 3, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)

    # Create SmoothGrad with configurable parameters
    smoothgrad = SmoothGrad(
        model,
        device=device,
        num_samples=20,  # Reduced for demo
        noise_level=0.1,
    )

    images = torch.randn(2, 3, 224, 224).to(device).reshape(2, -1)
    class_labels = torch.tensor([0, 1]).to(device)

    print("Generating smoothed saliency (20 samples with noise)...")
    results = smoothgrad.explain(images.reshape(2, 3, 224, 224), class_labels)

    print(f"SmoothGrad Results:")
    print(f"  saliency shape: {results['saliency'].shape}")
    print(f"  smoothness metric: saliency std={results['saliency'].std():.6f}")

    # Compare noise levels
    print(f"\n  Processing complete for {results['saliency'].shape[0]} samples")


# ============================================================================
# Example 4: Attention Map Extraction
# ============================================================================


def example_attention_extraction() -> None:
    """Demonstrate attention map extraction."""
    print("\n" + "=" * 70)
    print("Example 4: Attention Map Extraction")
    print("=" * 70)

    from explainability.attention_maps import AttentionMapExtractor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model with attention
    class SimpleAttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
            self.classifier = nn.Linear(64, 10)

        def forward(self, x):
            # Conv2d: [B, 3, H, W] -> [B, 64, H, W]
            x = self.conv(x)

            # Reshape for attention: [B, 64, H, W] -> [B, H*W, 64]
            b, c, h, w = x.shape
            x = x.reshape(b, c, -1).permute(0, 2, 1)  # [B, H*W, 64]

            # Apply attention
            attn_out, attn_weights = self.attention(x, x, x)

            # Average pool
            x = attn_out.mean(dim=1)  # [B, 64]

            # Classify
            return self.classifier(x)

    model = SimpleAttentionModel().to(device)

    extractor = AttentionMapExtractor(model, device=device)

    # Find attention layers
    attention_layers = extractor._find_attention_layers()
    print(f"Found attention layers: {attention_layers}")

    # Extract attention
    images = torch.randn(2, 3, 64, 64).to(device)

    try:
        attention_dict = extractor.explain(images)

        print(f"\nAttention Extraction Results:")
        for layer_name, attn in attention_dict.items():
            print(f"  {layer_name}:")
            print(f"    Shape: {attn.shape}")
            if attn.ndim > 2:
                print(f"    Mean attention: {attn.mean().item():.4f}")
    except Exception as e:
        print(f"  Note: {e}")


# ============================================================================
# Example 5: Unified Explainer Interface
# ============================================================================


def example_unified_explainer() -> None:
    """Demonstrate unified Explainer interface."""
    print("\n" + "=" * 70)
    print("Example 5: Unified Explainer Interface")
    print("=" * 70)

    from explainability.explainer import Explainer, ExplainerFactory

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dummy multi-task model
    class MultiTaskModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 10),
            )
            self.seg_head = nn.Conv2d(64, 21, kernel_size=1)

        def forward(self, x):
            shared = self.shared(x)
            cls_out = self.cls_head(shared)
            seg_out = self.seg_head(shared)
            return {"classification": cls_out, "segmentation": seg_out}

    model = MultiTaskModel().to(device)

    # Create explainer
    explainer = Explainer(model, device=device)

    print(f"Available methods: {explainer.list_available_methods()}")
    print(f"Supported tasks: {explainer.supported_tasks()}")

    # Explain classification
    images = torch.randn(2, 3, 128, 128).to(device)
    class_labels = torch.tensor([0, 1]).to(device)
    seg_masks = torch.randint(0, 21, (2, 128, 128)).to(device)

    print("\nExplaining classification...")
    try:
        cls_explain = explainer.explain_classification(
            images, class_labels, method="saliency"
        )
        print(f"  Result keys: {list(cls_explain.keys())}")
        print(f"  Saliency shape: {cls_explain['saliency'].shape}")
    except Exception as e:
        print(f"  Note: {e}")

    print("\nExplaining segmentation...")
    try:
        seg_explain = explainer.explain_segmentation(
            images, seg_masks, method="saliency"
        )
        print(f"  Result keys: {list(seg_explain.keys())}")
        print(f"  Magnitude shape: {seg_explain['magnitude'].shape}")
    except Exception as e:
        print(f"  Note: {e}")

    # Factory pattern
    print("\nUsing ExplainerFactory...")
    factory_explainer = ExplainerFactory.for_classification(model, device)
    print(f"  Created classification explainer: {type(factory_explainer)}")


# ============================================================================
# Example 6: Integrated Gradients
# ============================================================================


def example_integrated_gradients() -> None:
    """Demonstrate Integrated Gradients."""
    print("\n" + "=" * 70)
    print("Example 6: Integrated Gradients")
    print("=" * 70)

    from explainability.saliency import IntegratedGradients

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    ig = IntegratedGradients(model, device=device, num_steps=15)

    images = torch.randn(2, 784).to(device)
    class_labels = torch.tensor([0, 1]).to(device)
    baseline = torch.zeros_like(images)

    print("Computing integrated gradients (15 steps)...")
    results = ig.explain(images, class_labels, baseline=baseline)

    print(f"Integrated Gradients Results:")
    print(f"  attributions shape: {results['attributions'].shape}")
    print(f"  attributions range: [{results['attributions'].min():.6f}, {results['attributions'].max():.6f}]")
    print(f"  top-5 attributed features (sample 0):")

    attr = results["attributions"][0]
    top_idx = torch.topk(attr.flatten(), k=5).indices
    for i, idx in enumerate(top_idx):
        print(f"    {i+1}. Feature {idx.item()}: {attr.flatten()[idx].item():.6f}")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    print("\nPhase 4 Integration Examples")
    print("Demonstrating explainability and interpretability methods")

    # Run examples
    example_grad_cam_classification()
    example_vanilla_saliency()
    example_smooth_grad()
    example_attention_extraction()
    example_unified_explainer()
    example_integrated_gradients()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
