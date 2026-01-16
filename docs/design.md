# Design Philosophy & Extensibility

## Core Principles

### 1. Modularity Over Convenience
Each component solves exactly one problem. Combining components is explicit. We avoid implicit dependencies and framework magic.

```python
# Good: Explicit composition
backbone = CNNBackbone(depth=50)
attention = CBAM(in_channels=2048)
decoder = UNetDecoder(backbone_channels=[256, 512, 1024, 2048])

# Bad: Hidden dependencies
model = AutomaticVisualizationPipeline(task="segmentation")  # What's inside?
```

### 2. Transparency Over Abstraction
Code should be readable and debuggable. When in doubt, be verbose.

```python
# Good: Clear intent
for images, labels in val_loader:
    outputs = model(images)
    logits = outputs  # No hidden processing
    loss = criterion(logits, labels)
    
# Bad: Hidden transformations
for batch in smart_loader:  # Does it shuffle? Preprocess? Normalize?
    loss = trainer.step(batch)  # Where's the model forward pass?
```

### 3. Composability Over Inheritance
Use composition (objects working together) instead of deep class hierarchies.

```python
# Good: Composition
class Model:
    def __init__(self, backbone, attention=None, decoder=None):
        self.backbone = backbone
        self.attention = attention
        self.decoder = decoder
    
    def forward(self, x):
        x = self.backbone(x)
        if self.attention:
            x = self.attention(x)
        if self.decoder:
            x = self.decoder(x)
        return x

# Bad: Inheritance chains
class SegmentationModel(BackboneModel):
    class with CustomAttentionVariant(AttentionModel):
        class with ReversibleDecoder(DecoderModel):
            ...  # Multiple inheritance, method resolution order nightmare
```

### 4. Explicit Configuration Over Magic
All behavior should be controlled by explicit parameters. No hidden defaults.

```python
# Good: All parameters explicit
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    max_epochs=100,
    early_stopping_patience=10,
    mixed_precision=False,
    gradient_clip_value=1.0,
)

# Bad: Implicit defaults
trainer = Trainer(model)  # What's the learning rate? When does it stop?
```

### 5. Determinism by Default
Reproducibility is not optional. All randomness is controllable.

```python
# Good: Single call enables reproducibility
set_seed(42)
# Now all training is reproducible

# Bad: Scattered seed calls
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# What about CUDA? What about cuDNN?
```

## Architectural Patterns

### Pattern 1: Feature Map Hierarchy (Backbones)

All backbones return a **consistent spatial pyramid** of features:

```
Input: [B, 3, H, W]
  ↓
Stage 1: [B, C1, H/4, W/4]
Stage 2: [B, C2, H/8, W/8]
Stage 3: [B, C3, H/16, W/16]
Stage 4: [B, C4, H/32, W/32]
```

This allows:
- Generic decoders that work with any backbone
- Attention modules that process the same spatial dimensions
- Skip connections in segmentation models
- Multi-scale loss computation

**Why this pattern?**
- Matches natural receptive field growth
- Compatible with modern vision architectures (ResNet, ViT, DeiT)
- Enables hopping to any intermediate layer without rewriting code

### Pattern 2: Hook-Based Feature Extraction (Explainability)

Instead of modifying models, hooks capture activations:

```python
activations = []

def hook_fn(module, input, output):
    activations.append(output)

handle = layer.register_forward_hook(hook_fn)
# ... forward pass ...
handle.remove()
```

**Why this pattern?**
- Works with any pre-trained model (ResNet, ViT, EfficientNet, etc.)
- Model code is never modified
- Multiple hooks can coexist
- Clean separation between model and interpretation

### Pattern 3: Stateful Metrics (Evaluation)

Metrics maintain running accumulators that are explicitly reset, updated, and computed:

```python
metrics = ClassificationMetrics(num_classes=10)

for batch in dataset:
    outputs = model(batch["image"])
    metrics.update(outputs.argmax(1), batch["label"])

final_scores = metrics.compute()
metrics.reset()
```

**Why this pattern?**
- Deterministic: No randomness in metric computation
- Transparent: Can inspect state at any time
- Efficient: Single forward pass, accumulators updated incrementally
- Auditable: Easy to verify correctness

### Pattern 4: Task Auto-Detection (Training)

Tasks are distinguished by batch keys, not explicit flags:

```python
# Classification: batch has "image" and "label"
batch = {"image": x, "label": y}
outputs = model(x)  # Returns logits [B, C]

# Segmentation: batch has "image" and "mask"
batch = {"image": x, "mask": m}
outputs = model(x)  # Returns logits [B, C, H, W]

# Multi-task: batch has all three
batch = {"image": x, "label": y, "mask": m}
outputs = model(x)  # Returns dict {"classification": logits_cls, "segmentation": logits_seg}
```

**Why this pattern?**
- Reduces boilerplate (no explicit task parameter)
- Easy to support mixed tasks in single batch
- Model responsibility is clear: look at output type, not a flag

## Extensibility Patterns

### Adding a New Backbone

```python
# 1. Create architecture that returns feature pyramid
class MyBackbone(nn.Module):
    def forward(self, x):
        # Return list of feature maps at different scales
        return [feat_s4, feat_s8, feat_s16, feat_s32]

# 2. Wrap with MultitaskModel
backbone = MyBackbone()
model = MultitaskModel(
    backbone=backbone,
    decoder=UNetDecoder(backbone.channels),
    num_classes_classification=10,
    num_classes_segmentation=21,
)

# 3. Use with Trainer
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    max_epochs=100,
)
trainer.train(train_loader, val_loader)
```

### Adding a New Explainability Method

```python
# 1. Inherit from base
class MyExplainabilityMethod(BaseExplainer):
    def explain(self, inputs, model, target_class):
        # Compute your attribution method
        # Return [B, H, W] heatmap or [B, C, H, W] multi-class
        attributions = ...
        return attributions.detach().cpu().numpy()

# 2. Register in Explainer
explainer = Explainer(
    model=model,
    methods=["grad_cam", "my_method"],
)
explanations = explainer.explain(inputs, target_class=0)
# Returns dict with grad_cam and my_method results
```

### Adding a New Dataset

```python
# 1. Create dataset class
class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = glob(f"{root}/*.jpg")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        # Return dict with required keys
        return {
            "image": self.transform(image),
            "label": get_label(self.images[idx]),  # For classification
            "mask": get_mask(self.images[idx]),     # For segmentation
        }

# 2. Use with Trainer
train_dataset = MyDataset("data/train", transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

trainer.train(train_loader, val_loader)
```

### Adding a New Metric

```python
# 1. Inherit from base
class MyMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.tensor(0))
        self.register_buffer("sum", torch.tensor(0.0))
    
    def update(self, outputs, targets):
        # Accumulate
        self.sum += compute_metric(outputs, targets)
        self.count += 1
    
    def compute(self):
        # Return final scalar
        return self.sum / self.count
    
    def reset(self):
        self.count.zero_()
        self.sum.zero_()

# 2. Use in Evaluator
evaluator = Evaluator(model=model)
custom_metric = MyMetric()

for batch in dataset:
    outputs = model(batch["image"])
    custom_metric.update(outputs, batch["label"])

score = custom_metric.compute()
```

## Trade-offs & Philosophy

### Flexibility vs Simplicity
We prioritize flexibility. Single-use cases are solved simply; complex cases are possible.

```python
# Simple case: default training
trainer.train(train_loader, val_loader)

# Complex case: custom loss, multi-GPU, distributed training, etc.
# All possible because trainer doesn't hide the model
for epoch in range(max_epochs):
    for batch in train_loader:
        loss = compute_loss(model, batch, criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Speed vs Debuggability
We accept slower operations if they enable debugging. Use a profiler to find bottlenecks, don't guess.

```python
# We compute metrics explicitly every epoch (slower)
evaluator.evaluate(val_loader)

# Not batched across epochs (more memory overhead)
# But: You can inspect metrics at any epoch without recomputation
```

### Memory vs Convenience
Mixed precision and gradient accumulation are explicit choices, not automatic.

```python
# Good: You decide
trainer = Trainer(mixed_precision=True)

# Bad: Hidden memory consumption
trainer = Trainer()  # Uses how much memory? Nobody knows.
```

## Code Organization

**By responsibility:**
- `models/` - All architectures (no training logic)
- `training/` - All training logic (no model definitions)
- `evaluation/` - All evaluation logic (no model definitions)
- `explainability/` - All interpretation logic (no model definitions)
- `data/` - All dataset and transform logic

**By reusability:**
- Core utilities are pure functions or simple classes
- Complex workflows compose small pieces
- No "god classes" that do everything

**By testability:**
- Each module has clear inputs/outputs
- No global state or side effects
- Deterministic given fixed inputs
