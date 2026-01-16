"""Phase 2 Integration Example: Training System with Phase 1 Models.

This example demonstrates:
1. Creating generic datasets from image paths
2. Building augmentation pipelines for training and validation
3. Training Phase 1 models (classification, segmentation, multi-task)
4. Using mixed precision for efficient training
5. Checkpointing and early stopping
6. Accessing training history

Phase 2 builds on Phase 1 models without modification, adding:
- Generic data loading (ClassificationDataset, SegmentationDataset)
- Transform pipeline (Resize, Flips, Normalize, Compose)
- Unified Trainer supporting classification, segmentation, multi-task
- Mixed precision training utilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Phase 1 models and tasks
from models.backbones.cnn import CNN
from models.backbones.vit_spatial import ViTSpatial
from models.attention.cbam import CBAM
from models.encoder_decoder.unet_decoder import UNetDecoder
from models.multitask import MultiTaskModel

# Phase 2 data and training
from data.datasets import (
    ClassificationDataset, 
    DummyClassificationDataset,
    SegmentationDataset,
    DummySegmentationDataset,
)
from data.transforms import (
    get_train_transforms,
    get_val_transforms,
)
from training.trainer import Trainer


# ==============================================================================
# Example 1: Classification Training with Dummy Data
# ==============================================================================

def example_1_classification_training():
    """Training a CNN classifier using dummy data.
    
    Demonstrates:
    - DummyClassificationDataset for testing without files
    - Compose transform pipeline
    - Trainer with single loss function
    - Early stopping
    """
    print("\n" + "="*80)
    print("Example 1: Classification Training")
    print("="*80)
    
    # Create model
    model = CNN(
        num_classes=10,
        depth=2,
        width_multiplier=1.0,
        input_channels=3,
    )
    
    # Create dummy datasets
    train_dataset = DummyClassificationDataset(
        num_samples=50,
        num_classes=10,
        image_size=(224, 224),
        transforms=get_train_transforms(),
    )
    
    val_dataset = DummyClassificationDataset(
        num_samples=20,
        num_classes=10,
        image_size=(224, 224),
        transforms=get_val_transforms(),
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp_enabled=True,
        checkpoint_dir="./checkpoints_phase2/classification",
        seed=42,
    )
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        patience=2,
    )
    
    print(f"\nFinal Training Loss: {history['train_loss'][-1]:.6f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
    print("Classification training complete!")


# ==============================================================================
# Example 2: Segmentation Training with Dummy Data
# ==============================================================================

def example_2_segmentation_training():
    """Training a segmentation model using dummy data.
    
    Demonstrates:
    - DummySegmentationDataset with random image-mask pairs
    - Transforms applied to both image and mask consistently
    - Trainer for segmentation (outputs [B, C, H, W])
    """
    print("\n" + "="*80)
    print("Example 2: Segmentation Training")
    print("="*80)
    
    # Create backbone and decoder
    backbone = ViTSpatial(
        image_size=224,
        patch_size=16,
        num_layers=2,
        hidden_dim=256,
        num_heads=4,
    )
    
    decoder = UNetDecoder(
        encoder_channels=[256],  # Output from ViT
        num_classes=5,
        upsample_mode="bilinear",
    )
    
    # Create segmentation model
    class SegmentationModel(nn.Module):
        def __init__(self, backbone, decoder):
            super().__init__()
            self.backbone = backbone
            self.decoder = decoder
        
        def forward(self, x):
            features = self.backbone(x)  # [B, C, H/16, W/16]
            output = self.decoder(features)  # [B, num_classes, H, W]
            return output
    
    model = SegmentationModel(backbone, decoder)
    
    # Create dummy segmentation datasets
    train_dataset = DummySegmentationDataset(
        num_samples=50,
        num_classes=5,
        image_size=(224, 224),
        transforms=get_train_transforms(),
    )
    
    val_dataset = DummySegmentationDataset(
        num_samples=20,
        num_classes=5,
        image_size=(224, 224),
        transforms=get_val_transforms(),
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp_enabled=True,
        checkpoint_dir="./checkpoints_phase2/segmentation",
        seed=42,
    )
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        patience=2,
    )
    
    print(f"\nFinal Training Loss: {history['train_loss'][-1]:.6f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
    print("Segmentation training complete!")


# ==============================================================================
# Example 3: Multi-Task Training
# ==============================================================================

def example_3_multitask_training():
    """Training a multi-task model (classification + segmentation).
    
    Demonstrates:
    - MultiTaskModel from Phase 1 (outputs dict with two tasks)
    - Loss function dict for multi-task learning
    - Trainer handling dict outputs
    """
    print("\n" + "="*80)
    print("Example 3: Multi-Task Training (Classification + Segmentation)")
    print("="*80)
    
    # Create model
    backbone = CNN(num_classes=256, depth=3)
    decoder = UNetDecoder(encoder_channels=[256], num_classes=5)
    
    model = MultiTaskModel(
        backbone=backbone,
        decoder=decoder,
        num_classes=10,  # Classification classes
        num_segments=5,  # Segmentation classes
    )
    
    # Create dummy datasets with both labels and masks
    class MultiTaskDataset(DummyClassificationDataset):
        def __getitem__(self, idx):
            sample = super().__getitem__(idx)
            # Add mask for segmentation task
            mask = torch.randint(0, 5, (224, 224))
            sample["mask"] = mask
            return sample
    
    train_dataset = MultiTaskDataset(
        num_samples=50,
        num_classes=10,
        image_size=(224, 224),
        transforms=get_train_transforms(),
    )
    
    val_dataset = MultiTaskDataset(
        num_samples=20,
        num_classes=10,
        image_size=(224, 224),
        transforms=get_val_transforms(),
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Setup training with loss dict for multi-task
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = {
        "classification": nn.CrossEntropyLoss(),
        "segmentation": nn.CrossEntropyLoss(),
    }
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp_enabled=True,
        checkpoint_dir="./checkpoints_phase2/multitask",
        seed=42,
    )
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        patience=2,
    )
    
    print(f"\nFinal Training Loss: {history['train_loss'][-1]:.6f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
    print("Multi-task training complete!")


# ==============================================================================
# Example 4: Custom Dataset with File Paths (Classification)
# ==============================================================================

def example_4_custom_classification_dataset():
    """Demonstrating ClassificationDataset with custom image paths.
    
    In real usage, you would:
    - Collect image paths from disk
    - Assign class labels
    - Pass to ClassificationDataset
    - Use with Trainer
    
    This example shows the API without requiring actual files.
    """
    print("\n" + "="*80)
    print("Example 4: Custom Classification Dataset (API Demo)")
    print("="*80)
    
    print("\nClassificationDataset API:")
    print("  dataset = ClassificationDataset(")
    print("      image_paths=['path/to/img1.jpg', 'path/to/img2.jpg', ...],")
    print("      labels=[0, 1, ...],  # Class indices")
    print("      transforms=get_train_transforms(),")
    print("  )")
    print("\n  # DataLoader will yield dict samples:")
    print("  for batch in DataLoader(dataset):")
    print("      # batch['image'] -> [B, 3, H, W] tensor")
    print("      # batch['label'] -> [B] long tensor")
    print("\nRefer to data/datasets.py for full documentation.")


# ==============================================================================
# Example 5: Transform Pipeline Demonstration
# ==============================================================================

def example_5_transforms_demo():
    """Demonstrating the transform pipeline.
    
    Shows:
    - Individual transforms (Resize, Flip, Normalize)
    - Compose for chaining transforms
    - Separate train and val pipelines
    """
    print("\n" + "="*80)
    print("Example 5: Transform Pipeline")
    print("="*80)
    
    # Create sample
    image = torch.rand(3, 512, 512)
    mask = torch.randint(0, 5, (512, 512))
    sample = {"image": image, "mask": mask}
    
    print(f"\nOriginal sample shapes:")
    print(f"  image: {sample['image'].shape}")
    print(f"  mask: {sample['mask'].shape}")
    
    # Apply training transforms
    train_transforms = get_train_transforms(image_size=(224, 224))
    transformed = train_transforms(sample)
    
    print(f"\nAfter training transforms:")
    print(f"  image: {transformed['image'].shape}, range: [{transformed['image'].min():.3f}, {transformed['image'].max():.3f}]")
    print(f"  mask: {transformed['mask'].shape}, unique: {transformed['mask'].unique().tolist()}")
    
    # Show that validation transforms don't do random flips
    val_transforms = get_val_transforms(image_size=(224, 224))
    print(f"\nValidation transforms exclude augmentation:")
    print(f"  - get_train_transforms: Resize + RandomFlips + Normalize")
    print(f"  - get_val_transforms: Resize + Normalize (no augmentation)")


# ==============================================================================
# Example 6: Trainer Features
# ==============================================================================

def example_6_trainer_features():
    """Demonstrating key Trainer features.
    
    Shows:
    - Mixed precision training
    - Checkpointing and loading
    - Early stopping
    - Training history access
    - Seed control for reproducibility
    """
    print("\n" + "="*80)
    print("Example 6: Trainer Features")
    print("="*80)
    
    print("\nTrainer Key Features:")
    print("  1. Mixed Precision Training")
    print("     - Automatic mixed precision (AMP) on GPU")
    print("     - Gradient scaling and unscaling")
    print("     - CPU fallback with identity scaler")
    print("     - Parameters: amp_enabled=True")
    
    print("\n  2. Checkpointing")
    print("     - Auto-saves best model during validation")
    print("     - Manual save/load with full state")
    print("     - Methods: save_checkpoint(path), load_checkpoint(path)")
    
    print("\n  3. Early Stopping")
    print("     - Stops training if validation loss doesn't improve")
    print("     - Parameters: patience=N (epochs without improvement)")
    print("     - Example: fit(..., patience=3)")
    
    print("\n  4. Training History")
    print("     - Tracks train_loss and val_loss per epoch")
    print("     - Access via get_training_history()")
    print("     - Dict with keys: ['train_loss', 'val_loss']")
    
    print("\n  5. Reproducibility")
    print("     - Random seed control: seed parameter")
    print("     - Ensures deterministic results")
    
    print("\n  6. Device Support")
    print("     - Auto-detects GPU (cuda) vs CPU")
    print("     - Fallback to CPU if cuda not available")
    
    print("\n  7. Multi-Task Support")
    print("     - loss_fn as dict for multi-task models")
    print("     - Automatic loss aggregation")


# ==============================================================================
# Main: Run All Examples
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHASE 2 INTEGRATION EXAMPLES: Training System")
    print("="*80)
    print("\nThis script demonstrates Phase 2 components:")
    print("- Generic data loading (ClassificationDataset, SegmentationDataset)")
    print("- Transform pipeline (Resize, Flips, Normalize)")
    print("- Unified Trainer for Phase 1 models")
    print("- Mixed precision training")
    print("- Checkpointing and early stopping")
    
    # Run examples
    example_1_classification_training()
    example_2_segmentation_training()
    example_3_multitask_training()
    example_4_custom_classification_dataset()
    example_5_transforms_demo()
    example_6_trainer_features()
    
    print("\n" + "="*80)
    print("All examples complete!")
    print("="*80)
    print("\nPhase 2 Implementation Summary:")
    print("✓ data/datasets.py - Generic dataset classes")
    print("✓ data/transforms.py - Composable transform pipeline")
    print("✓ training/mixed_precision.py - AMP utilities")
    print("✓ training/trainer.py - Unified trainer for Phase 1 models")
    print("\nNext Steps (Phase 3):")
    print("- Evaluation metrics (accuracy, IoU, F1, etc.)")
    print("- Statistical significance testing")
    print("- Experiment runners with config system")
    print("="*80 + "\n")
