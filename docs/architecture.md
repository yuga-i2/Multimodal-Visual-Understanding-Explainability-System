# System Architecture

## Overview

The Vision Understanding Platform is organized as a layered ML system with four independent, composable components:

1. **Models**: Architectures for vision tasks (CNN, ViT, U-Net, multi-task)
2. **Training**: Unified trainer with mixed precision, determinism, and checkpointing
3. **Evaluation**: Task-aware metrics and experiment orchestration
4. **Explainability**: Gradient-based and attention-based interpretation methods

Each layer is completely decoupled. You can use models without training, training without evaluation, or explainability on any external model.

## System Block Diagram

```
┌─────────────────────────────────────────────────────────┐
│ Data Pipeline                                           │
│ (Images, labels, masks)                                 │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Models (Architectures)                                  │
│ • Backbones: CNN, ViT, ViT+Spatial                      │
│ • Components: Attention (CBAM), Decoders (U-Net)        │
│ • Outputs: Logits [B,C] or [B,C,H,W] or mixed dict     │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Training System                                         │
│ • Unified Trainer (classification, segmentation, multi) │
│ • Mixed Precision (AMP) with gradient scaling           │
│ • Dataset abstractions and augmentation pipeline        │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Evaluation & Experimentation                            │
│ • Metrics: Task-specific accumulators                   │
│ • Evaluator: Generic evaluation engine                  │
│ • Experiment: Orchestrates training + evaluation        │
│ • Reproducibility: Seed control, CUDNN determinism     │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Explainability & Interpretability                       │
│ • Methods: Grad-CAM, Saliency, SmoothGrad, IntGrad    │
│ • Attention extraction (transformer weights)            │
│ • Unified interface: Auto-selects method                │
│ • Hook-based: Zero model modifications                  │
└─────────────────────────────────────────────────────────┘
```

## Component Breakdown

### Models Layer

**Backbones** return consistent spatial feature maps [B, C, H, W]:
- **CNN** - Convolutional residual backbone (depth/width configurable)
- **ViT** - Vision Transformer with patch embeddings
- **ViT Spatial** - ViT that preserves spatial structure (no global reduction)

**Attention Modules** are optional, composable components:
- **CBAM** - Channel + Spatial Attention Module (squeeze-and-excitation style)

**Encoder-Decoder** for segmentation:
- **U-Net Decoder** - Generic decoder accepting arbitrary backbones
- Progressive upsampling with skip connections
- Works with any backbone (CNN, ViT, hybrid)

**Multi-Task Model** wraps all components:
- Classification head (global pooling + FC)
- Segmentation head (1×1 conv + upsampling)
- Flexible composition (can disable any component)

### Training Layer

**Trainer** abstracts training loops:
- Task-agnostic (classification, segmentation, multi-task auto-detected)
- Automatic mixed precision (AMP) with gradient scaling
- Gradient clipping and adaptive learning
- Validation-based checkpointing and early stopping

**Mixed Precision Manager**:
- Wraps `torch.cuda.amp.autocast` and `GradScaler`
- Falls back to FP32 on CPU
- Transparent to user

**Dataset Abstractions**:
- `ClassificationDataset` - File-based image + label pairs
- `SegmentationDataset` - File-based image + mask pairs
- `DummyClassificationDataset` - Synthetic data for testing
- `DummySegmentationDataset` - Synthetic data for testing

**Transform Pipeline**:
- Composable: `Resize`, `RandomFlip`, `Normalize`
- Consistent application to image and mask (for segmentation)
- Separate train/val transforms

### Evaluation Layer

**Metrics** follow stateful accumulator pattern:
- `ClassificationMetrics` - Accuracy, top-k accuracy
- `SegmentationMetrics` - Per-class and mean IoU, per-class and mean Dice
- Explicit reset/update/compute interface

**Evaluator** - Generic evaluation engine:
- Accepts any nn.Module
- Task auto-detection from batch keys
- No gradient computation (memory efficient)
- GPU-aware but CPU-safe

**Experiment Orchestrator**:
- Coordinates Trainer + Evaluator
- History tracking per epoch
- Checkpoint management
- Reproducibility control

**Reproducibility Utilities**:
- `set_seed()` - Controls Python, NumPy, PyTorch, CUDA
- `enable_cudnn_determinism()` - GPU determinism control

### Explainability Layer

**Grad-CAM**:
- Gradient-weighted class activation maps
- Works for classification and segmentation
- Layer-wise target selection
- Hook-based (zero model modification)

**Saliency Methods**:
- Vanilla Saliency - Input gradient magnitude
- SmoothGrad - Noise-averaged gradients
- Integrated Gradients - Path-based attribution

**Attention Extraction**:
- Extracts multi-head transformer attention weights
- Aggregates across heads and layers
- Works with standard `nn.MultiheadAttention`

**Unified Explainer Interface**:
- Single API for all methods
- Auto-selects method based on model architecture
- Auto-detects task (classification vs segmentation)
- Batch processing with GPU support

## Data Flow

### Classification Task

```
Input: batch["image"] [B, 3, H, W]
       batch["label"] [B]

→ Backbone: Extract features [B, C, H, W]
→ Optional Attention: Refine features [B, C, H, W]
→ Classification Head: Pool + Linear [B, num_classes]
→ Trainer: Compute CrossEntropyLoss([B, num_classes], [B])
→ Evaluator: Compute accuracy, top-k accuracy
→ Explainer: Generate CAM, saliency, etc.
```

### Segmentation Task

```
Input: batch["image"] [B, 3, H, W]
       batch["mask"] [B, H, W] (class indices)

→ Backbone: Extract multi-scale features
→ Decoder: Progressive upsampling with skips [B, num_classes, H, W]
→ Trainer: Compute CrossEntropyLoss([B, num_classes, H, W], [B, H, W])
→ Evaluator: Compute per-class and mean IoU, Dice
→ Explainer: Generate CAM, saliency per-class
```

### Multi-Task

```
Input: batch["image"] [B, 3, H, W]
       batch["label"] [B]
       batch["mask"] [B, H, W]

→ Backbone: Extract features [B, C, H, W]
→ Trainer: Compute classification + segmentation losses (weighted sum)
→ Evaluator: Compute both metric sets
→ Explainer: Generate explanations for both heads
```

## Key Design Decisions

**Why Pure PyTorch?**
Core logic is transparent and debuggable. No dependency on framework abstractions or external explainability libraries.

**Why Hook-Based Explainability?**
Zero modifications to models. Any pre-trained PyTorch model can be explained post-hoc.

**Why Modular Layers?**
Each layer solves a distinct problem and can be used independently. Composition is explicit, not automatic.

**Why Stateful Metrics?**
Explicit reset/update/compute makes metric computation transparent and auditable. No hidden state.

**Why Task Auto-Detection?**
Classification, segmentation, and multi-task are distinguished by batch keys, not explicit configuration. Reduces boilerplate.

**Why Deterministic Training?**
Reproducibility is critical for research and production. Full seed control is built-in, not optional.
