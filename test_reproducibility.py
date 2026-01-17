"""Reproducibility verification: Run training twice with same seed and compare."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Seed everything for reproducibility
SEED = 42

def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_training(run_name: str) -> dict:
    """Run classification training and return metrics."""
    print(f"\n{'='*70}")
    print(f"{run_name}")
    print(f"{'='*70}")
    
    set_seeds(SEED)
    
    from models.backbones.cnn import create_cnn_backbone
    from models.multitask import MultiTaskModel
    from data.datasets import DummyClassificationDataset
    from data.transforms import get_train_transforms, get_val_transforms
    from training.trainer import Trainer
    
    # Create model with same architecture
    backbone = create_cnn_backbone(in_channels=3, base_channels=64, depth="small")
    model = MultiTaskModel(
        backbone=backbone,
        backbone_out_channels=512,
        num_classes=10,
        include_segmentation_head=False,
        attention_module=None,
    )
    
    # Create dummy datasets (same seeds used above)
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
    loss_fn = {"classification": nn.CrossEntropyLoss()}
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
        amp_enabled=False,
    )
    
    # Train for 3 epochs
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        patience=10,
    )
    
    # Extract metrics
    metrics = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "model_state": model.state_dict(),
    }
    
    print(f"\nTrain loss: {metrics['train_loss']}")
    print(f"Val loss:   {metrics['val_loss']}")
    
    return metrics

def compare_reproducibility(metrics1: dict, metrics2: dict) -> bool:
    """Compare metrics from two runs."""
    print(f"\n{'='*70}")
    print("REPRODUCIBILITY VERIFICATION RESULTS")
    print(f"{'='*70}")
    
    # Compare losses
    train_l1 = metrics1["train_loss"]
    train_l2 = metrics2["train_loss"]
    val_l1 = metrics1["val_loss"]
    val_l2 = metrics2["val_loss"]
    
    print(f"\nTrain Loss:")
    print(f"  Run 1: {train_l1}")
    print(f"  Run 2: {train_l2}")
    
    train_match = all(
        abs(l1 - l2) < 1e-5
        for l1, l2 in zip(train_l1, train_l2)
    )
    print(f"  Match: {'✓ YES' if train_match else '✗ NO'}")
    
    print(f"\nVal Loss:")
    print(f"  Run 1: {val_l1}")
    print(f"  Run 2: {val_l2}")
    
    val_match = all(
        abs(l1 - l2) < 1e-5
        for l1, l2 in zip(val_l1, val_l2)
    )
    print(f"  Match: {'✓ YES' if val_match else '✗ NO'}")
    
    # Compare model weights
    print(f"\nModel State Dicts:")
    state1 = metrics1["model_state"]
    state2 = metrics2["model_state"]
    
    all_match = True
    for key in state1.keys():
        if key not in state2:
            print(f"  {key}: Missing in run 2")
            all_match = False
            continue
            
        diff = (state1[key] - state2[key]).abs().max().item()
        if diff > 1e-5:
            print(f"  {key}: Max diff = {diff:.2e} ✗")
            all_match = False
        else:
            print(f"  {key}: Identical ✓")
    
    overall = train_match and val_match and all_match
    
    print(f"\n{'='*70}")
    if overall:
        print("✓ REPRODUCIBILITY VERIFIED - Deterministic execution confirmed")
    else:
        print("✗ REPRODUCIBILITY FAILED - Non-deterministic behavior detected")
    print(f"{'='*70}")
    
    return overall

if __name__ == "__main__":
    print("\nREPRODUCIBILITY TEST: Running classification training 2x with seed=42")
    
    # Run 1
    metrics1 = run_training("RUN 1: Classification Training with seed=42")
    
    # Run 2
    metrics2 = run_training("RUN 2: Classification Training with seed=42")
    
    # Compare
    reproducible = compare_reproducibility(metrics1, metrics2)
    
    # Exit code
    exit(0 if reproducible else 1)
