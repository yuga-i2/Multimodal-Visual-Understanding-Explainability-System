# Implementation Reference

## Module Inventory

### Models (`models/`)

#### Backbones (`models/backbones/`)

**CNNBackbone** - Convolutional residual network
```python
from models.backbones import CNNBackbone

backbone = CNNBackbone(depth=50, pretrained=True)
# Output on any input [B, 3, H, W]:
# [B, 256, H/4, W/4]
# [B, 512, H/8, W/8]
# [B, 1024, H/16, W/16]
# [B, 2048, H/32, W/32]

# Supports: depth in [18, 34, 50, 101, 152]
# Has pretrained ImageNet weights available
```

**VisionTransformer (ViT)** - Patch-based transformer for vision
```python
from models.backbones import VisionTransformer

vit = VisionTransformer(
    image_size=224,
    patch_size=16,
    num_layers=12,
    hidden_dim=768,
    num_heads=12,
    mlp_dim=3072,
)
# Input: [B, 3, H, W]
# Output: [B, 768, H/16, W/16]  (spatial pyramid from intermediate layers)

# Supports ImageNet-21k pretrained weights
# Compatible with CLIP and other pre-trained ViT checkpoints
```

**ViTSpatial** - ViT that preserves spatial structure
```python
from models.backbones import ViTSpatial

vit_spatial = ViTSpatial(
    image_size=224,
    patch_size=16,
    num_layers=12,
)
# Like ViT but doesn't reduce to 1D sequence
# Output: [B, 768, 14, 14] (spatial dims preserved)
# Better for segmentation, explainability

# Same pretrained weight compatibility as ViT
```

#### Attention Modules (`models/attention/`)

**CBAM** - Channel and Spatial Attention Module
```python
from models.attention import CBAM

cbam = CBAM(in_channels=2048, reduction=16)
# Input: [B, C, H, W]
# Output: [B, C, H, W] (refined features)

# Stacks sequentially:
# 1. ChannelAttention: [B, C, H, W] → [B, C, 1, 1] → multiply
# 2. SpatialAttention: [B, C, H, W] → [B, 1, H, W] → multiply

# Can be inserted after any backbone layer
```

#### Decoders (`models/encoder_decoder/`)

**UNetDecoder** - Progressive upsampling with skip connections
```python
from models.encoder_decoder import UNetDecoder

decoder = UNetDecoder(
    backbone_channels=[256, 512, 1024, 2048],
    output_channels=21,  # num classes for segmentation
)
# Input: List of feature maps from backbone
#   [feat_s4, feat_s8, feat_s16, feat_s32]
# Output: [B, 21, H, W] (segmentation logits)

# Works with any backbone (CNN, ViT, hybrid)
# Efficient: minimal convolutions
# Supports deeplab-style atrous convolutions
```

#### Multi-Task Model

**MultitaskModel** - Unified classification + segmentation
```python
from models import MultitaskModel

model = MultitaskModel(
    backbone=backbone,
    attention=cbam,  # optional
    decoder=decoder,  # optional
    num_classes_classification=10,
    num_classes_segmentation=21,
)

# Forward pass auto-detects task from batch keys:
# If batch has "label": classification branch active
# If batch has "mask": segmentation branch active
# If both: both branches active (multi-task)

output = model(images)
# Returns:
# - Classification case: [B, 10] logits
# - Segmentation case: [B, 21, H, W] logits
# - Multi-task: {"classification": [B, 10], "segmentation": [B, 21, H, W]}
```

### Training (`training/`)

**Trainer** - Unified training engine
```python
from training import Trainer

trainer = Trainer(
    model=model,
    criterion=criterion,  # torch.nn.Module
    optimizer=optimizer,  # torch.optim.Optimizer
    max_epochs=100,
    validation_freq=1,     # Validate every N epochs
    early_stopping_patience=10,
    mixed_precision=False, # Use AMP
    gradient_clip_value=1.0,
    device="cuda",
)

# Train on multiple tasks automatically
trainer.train(train_loader, val_loader)

# Returns history dict:
# {"train_loss": [...], "val_loss": [...], "val_metrics": {...}}
```

**MixedPrecisionManager** - Automatic mixed precision
```python
from training import MixedPrecisionManager

amp = MixedPrecisionManager(enabled=True)

for batch in loader:
    with amp.autocast():
        loss = model(batch)
    amp.backward(loss)
    optimizer.step()

# Handles:
# - torch.cuda.amp.autocast for FP16 computation
# - torch.cuda.amp.GradScaler for gradient scaling
# - Fallback to FP32 on CPU
# - All cuDNN behavior
```

### Data (`data/`)

**Datasets**

```python
from data import ClassificationDataset, SegmentationDataset

# File-based datasets
train_dataset = ClassificationDataset(
    root="data/train",
    split="train",
    transform=transform,
)

seg_dataset = SegmentationDataset(
    images_dir="data/train/images",
    masks_dir="data/train/masks",
    transform=transform,
)

# Synthetic datasets for testing
from data import DummyClassificationDataset, DummySegmentationDataset

dummy = DummyClassificationDataset(
    num_samples=100,
    image_size=(224, 224),
    num_classes=10,
)
```

**Transforms**

```python
from data import Transforms

train_transforms = Transforms(
    resize=224,
    augment=True,
    normalize=True,
)

val_transforms = Transforms(
    resize=224,
    augment=False,
    normalize=True,
)

# Composable: each step is independent
# For segmentation: applied to both image and mask consistently
# Includes: resize, random flip, color jitter, normalization
```

### Evaluation (`evaluation/`)

**Metrics**

```python
from evaluation import ClassificationMetrics, SegmentationMetrics

# Classification
clf_metrics = ClassificationMetrics(num_classes=10)
clf_metrics.update(predictions, targets)
scores = clf_metrics.compute()
# Returns: {"accuracy": 0.95, "top5_accuracy": 0.99}

# Segmentation
seg_metrics = SegmentationMetrics(num_classes=21)
seg_metrics.update(predictions, targets)
scores = seg_metrics.compute()
# Returns: {
#     "mean_iou": 0.65,
#     "class_iou": [0.7, 0.6, ...],
#     "mean_dice": 0.70,
#     "class_dice": [0.75, 0.65, ...],
# }

# Reset for next epoch
clf_metrics.reset()
```

**Evaluator** - Generic evaluation engine
```python
from evaluation import Evaluator

evaluator = Evaluator(model=model, device="cuda")
metrics = evaluator.evaluate(val_loader)
# Auto-detects task from batch structure
# Returns dict of all metric results

# Works with any nn.Module, no modifications needed
```

**Experiment** - Orchestrates training + evaluation
```python
from evaluation import Experiment

experiment = Experiment(
    model=model,
    trainer=trainer,
    evaluator=evaluator,
    num_epochs=100,
    checkpoint_dir="checkpoints/",
)

history = experiment.run(train_loader, val_loader)
# Returns complete training history with all metrics
# Checkpoints saved automatically
```

**Reproducibility**

```python
from evaluation import set_seed, enable_cudnn_determinism

# Single call: fully deterministic training
set_seed(42)
enable_cudnn_determinism()

# Now all randomness is controlled:
# - Python: random module
# - NumPy: numpy.random
# - PyTorch: torch.manual_seed, torch.cuda.manual_seed_all
# - cuDNN: CUDNN_DETERMINISTIC=True, CUDNN_BENCHMARK=False
```

### Explainability (`explainability/`)

**BaseExplainer** - Abstract interface
```python
from explainability import BaseExplainer

class MyExplainer(BaseExplainer):
    def explain(self, inputs, model, target_class):
        # Compute attribution
        # Return [B, H, W] or [B, C, H, W] heatmap
        pass
```

**Grad-CAM** - Gradient-weighted class activation maps
```python
from explainability import GradCAM

gradcam = GradCAM(
    model=model,
    target_layer=model.backbone.layer4[-1],
)
heatmap = gradcam.explain(
    inputs,  # [B, 3, H, W]
    target_class=0,
    normalize=True,
)
# Returns [B, H, W] heatmap (0-1 range)

# Works for both classification and segmentation
# Batch processing GPU-efficient
```

**Saliency Methods** - Input gradient-based attribution
```python
from explainability import (
    VanillaSaliency,
    SmoothGrad,
    IntegratedGradients,
)

# Vanilla: Simple input gradient
saliency = VanillaSaliency(model)
attribution = saliency.explain(inputs, target_class=0)

# SmoothGrad: Gradient averaged over noise
smoothgrad = SmoothGrad(model, num_samples=50)
attribution = smoothgrad.explain(inputs, target_class=0)

# Integrated Gradients: Path-based attribution
intgrad = IntegratedGradients(model, num_steps=50)
attribution = intgrad.explain(inputs, target_class=0)

# All return [B, 3, H, W] (same shape as input)
```

**Attention Maps** - Transformer attention visualization
```python
from explainability import AttentionMaps

attention = AttentionMaps(model=model)
maps = attention.explain(inputs)
# Returns dict of attention weights per layer
# Aggregates across heads and normalizes
# Works with any transformer model
```

**Unified Explainer** - Single interface for all methods
```python
from explainability import Explainer

explainer = Explainer(
    model=model,
    methods=["grad_cam", "saliency", "smoothgrad", "attention"],
)

explanations = explainer.explain(inputs, target_class=0)
# Returns dict:
# {
#     "grad_cam": [B, H, W],
#     "saliency": [B, 3, H, W],
#     "smoothgrad": [B, 3, H, W],
#     "attention": {...},
# }

# Auto-detects model type (CNN, ViT, etc.)
# Auto-detects task (classification, segmentation)
# Falls back gracefully if method not applicable
```

## Usage Examples

### Example 1: Train Classification Model

```python
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from models import MultitaskModel
from models.backbones import CNNBackbone
from training import Trainer
from data import ClassificationDataset, Transforms

# Setup
set_seed(42)
device = "cuda"

# Data
train_transforms = Transforms(resize=224, augment=True)
val_transforms = Transforms(resize=224, augment=False)
train_dataset = ClassificationDataset("data/train", transform=train_transforms)
val_dataset = ClassificationDataset("data/val", transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model
backbone = CNNBackbone(depth=50, pretrained=True)
model = MultitaskModel(
    backbone=backbone,
    num_classes_classification=10,
).to(device)

# Training
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    max_epochs=50,
    device=device,
)
history = trainer.train(train_loader, val_loader)
```

### Example 2: Segmentation with ViT

```python
from models import MultitaskModel
from models.backbones import ViTSpatial
from models.encoder_decoder import UNetDecoder
from data import SegmentationDataset

# ViT backbone with spatial structure
backbone = ViTSpatial(image_size=224)

# U-Net decoder for upsampling
decoder = UNetDecoder(
    backbone_channels=backbone.get_channel_schedule(),
    output_channels=21,  # num classes
)

# Unified model
model = MultitaskModel(
    backbone=backbone,
    decoder=decoder,
    num_classes_segmentation=21,
).to(device)

# Training (same as before)
trainer.train(train_loader, val_loader)
```

### Example 3: Explainability

```python
from explainability import Explainer

# Load trained model
model.eval()

# Create explainer
explainer = Explainer(
    model=model,
    methods=["grad_cam", "smoothgrad", "attention"],
)

# Explain batch
with torch.no_grad():
    images, labels = next(iter(val_loader))
    explanations = explainer.explain(
        images.to(device),
        target_class=0,
    )

# Access individual explanations
grad_cam = explanations["grad_cam"]        # [B, H, W]
smoothgrad = explanations["smoothgrad"]    # [B, 3, H, W]
attention = explanations["attention"]      # dict per layer
```

### Example 4: Multi-Task Learning

```python
# Model with both classification and segmentation heads
model = MultitaskModel(
    backbone=backbone,
    attention=cbam,
    decoder=decoder,
    num_classes_classification=10,
    num_classes_segmentation=21,
)

# Batch with both tasks
batch = {
    "image": images,  # [B, 3, H, W]
    "label": labels,  # [B] for classification
    "mask": masks,    # [B, H, W] for segmentation
}

# Forward pass
output = model(batch["image"])
# Returns: {"classification": [B, 10], "segmentation": [B, 21, H, W]}

# Trainer handles both losses automatically
trainer.train(train_loader, val_loader)
```

## Testing Utilities

```python
# Synthetic data for quick testing
from data import DummyClassificationDataset

dummy = DummyClassificationDataset(
    num_samples=10,
    image_size=(224, 224),
    num_classes=10,
)
dummy_loader = DataLoader(dummy, batch_size=2)

# Test forward pass
model.eval()
with torch.no_grad():
    for batch in dummy_loader:
        output = model(batch["image"])
        print(output.shape)

# Test with trainer
trainer.train(dummy_loader, dummy_loader)
```
