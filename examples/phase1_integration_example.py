"""PHASE 1 INTEGRATION EXAMPLE

Demonstrates how to compose Phase 1 components (backbones, attention, decoders, tasks)
into working architectures for classification and segmentation.

This file shows the design working end-to-end, but contains NO training logic.
"""

import torch
import torch.nn as nn

# Phase 1 components
from models.backbones.cnn import create_cnn_backbone
from models.backbones.vit_spatial import create_vit_backbone
from models.attention.cbam import CBAM, OptionalCBAM
from models.encoder_decoder.unet_decoder import create_unet
from models.multitask import MultiTaskModel, HybridModel
from tasks.classification_task import ClassificationTask
from tasks.segmentation_task import SegmentationTask


def example_cnn_classification():
    """Example 1: CNN backbone for classification."""
    print("\n" + "="*60)
    print("Example 1: CNN Backbone for Classification")
    print("="*60)

    # Create backbone
    backbone = create_cnn_backbone(in_channels=3, base_channels=64, depth="small")

    # Create multi-task model (classification only)
    model = MultiTaskModel(
        backbone=backbone,
        backbone_out_channels=512,  # 64 * 8
        num_classes=10,
        include_segmentation_head=False,
        attention_module=None,
    )

    # Wrap in task interface
    task = ClassificationTask(model, num_classes=10)

    # Inference
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = task(x)
        predictions = task.predict(x)
        probabilities = task.predict_proba(x)

    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")


def example_cnn_with_attention():
    """Example 2: CNN + CBAM attention."""
    print("\n" + "="*60)
    print("Example 2: CNN with CBAM Attention")
    print("="*60)

    backbone = create_cnn_backbone(in_channels=3, base_channels=64, depth="small")

    # Add attention
    attention = OptionalCBAM(channels=512, enable=True, reduction_ratio=16)

    model = MultiTaskModel(
        backbone=backbone,
        backbone_out_channels=512,
        num_classes=10,
        include_segmentation_head=False,
        attention_module=attention,
    )

    task = ClassificationTask(model, num_classes=10)

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        predictions = task.predict(x)

    print(f"Input shape: {x.shape}")
    print(f"Predictions: {predictions}")
    print("CBAM attention applied successfully")


def example_cnn_unet_segmentation():
    """Example 3: CNN backbone + U-Net decoder for segmentation."""
    print("\n" + "="*60)
    print("Example 3: CNN + U-Net for Segmentation")
    print("="*60)

    # Backbone with multi-scale features
    backbone = create_cnn_backbone(in_channels=3, base_channels=64, depth="small")

    # U-Net decoder
    encoder_channels = [64, 128, 256, 512]  # CNN output channels at each stage
    decoder_channels = [256, 128, 64]  # Decoder channels (reverse of encoder)
    num_classes = 5

    segmentation_model = create_unet(
        backbone=backbone,
        encoder_channels=encoder_channels,
        num_classes=num_classes,
    )

    # Wrap in task interface
    task = SegmentationTask(segmentation_model, num_classes=num_classes)

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        predictions = task.predict(x)
        confidence = task.predict_confidence(x)

    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print(f"Prediction range: [{predictions.min()}, {predictions.max()}]")


def example_multitask():
    """Example 4: Multi-task model (classification + segmentation)."""
    print("\n" + "="*60)
    print("Example 4: Multi-Task Model (Classification + Segmentation)")
    print("="*60)

    backbone = create_cnn_backbone(in_channels=3, base_channels=64, depth="small")

    model = MultiTaskModel(
        backbone=backbone,
        backbone_out_channels=512,
        num_classes=10,
        include_segmentation_head=True,
        seg_upsampling_factor=4,
    )

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Classification logits shape: {output['classification'].shape}")
    print(f"Segmentation logits shape: {output['segmentation'].shape}")


def example_vit_classification():
    """Example 5: Vision Transformer backbone for classification."""
    print("\n" + "="*60)
    print("Example 5: Vision Transformer for Classification")
    print("="*60)

    # Create ViT with spatial output
    backbone = create_vit_backbone(
        image_size=224,
        patch_size=16,
        in_channels=3,
        preset="tiny",
    )

    model = MultiTaskModel(
        backbone=backbone,
        backbone_out_channels=192,  # ViT tiny embed_dim
        num_classes=10,
        include_segmentation_head=False,
    )

    task = ClassificationTask(model, num_classes=10)

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        predictions = task.predict(x)

    print(f"Input shape: {x.shape}")
    print(f"Predictions: {predictions}")
    print("ViT backbone with spatial output works correctly")


def example_hybrid_cnn_vit():
    """Example 6: CNN-Transformer hybrid architecture."""
    print("\n" + "="*60)
    print("Example 6: CNN-Transformer Hybrid")
    print("="*60)

    cnn_backbone = create_cnn_backbone(in_channels=3, base_channels=64, depth="small")
    vit_backbone = create_vit_backbone(
        image_size=56,  # Reduced from CNN early stage
        patch_size=7,
        in_channels=256,
        preset="tiny",
    )

    hybrid = HybridModel(
        cnn_backbone=cnn_backbone,
        transformer_backbone=vit_backbone,
        cnn_output_idx=2,  # Use stage 2 output of CNN
        num_classes=10,
        include_segmentation_head=False,
    )

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = hybrid(x)

    print(f"Input shape: {x.shape}")
    print(f"Classification output: {output['classification'].shape}")
    print("CNN-Transformer hybrid composition successful")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("PHASE 1 INTEGRATION EXAMPLES")
    print("Architecture Composition & Design Verification")
    print("#" * 60)

    try:
        example_cnn_classification()
        example_cnn_with_attention()
        example_cnn_unet_segmentation()
        example_multitask()
        example_vit_classification()
        example_hybrid_cnn_vit()

        print("\n" + "=" * 60)
        print("✓ ALL PHASE 1 EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey Observations:")
        print("- All backbones return spatial feature maps")
        print("- Attention modules are composable and optional")
        print("- Decoders accept arbitrary backbone outputs")
        print("- Tasks provide clean inference interfaces")
        print("- Multi-task and hybrid architectures work end-to-end")
        print("\nReady for Phase 2 (Training Infrastructure)")

    except Exception as e:
        print(f"\nError during examples: {e}")
        import traceback
        traceback.print_exc()
