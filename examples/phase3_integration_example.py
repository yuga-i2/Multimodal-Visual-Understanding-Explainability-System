"""Phase 3 Integration Examples.

Demonstrates usage of reproducibility, metrics, evaluator, and experiment
with Phase 1 models and Phase 2 training infrastructure.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

from evaluation.reproducibility import set_seed, enable_cudnn_determinism
from evaluation.metrics import ClassificationMetrics, SegmentationMetrics
from evaluation.evaluator import Evaluator
from evaluation.experiment import Experiment
from training.trainer import Trainer

# ============================================================================
# Example 1: Classification Model Evaluation
# ============================================================================


def example_classification_evaluation() -> None:
    """Evaluate a classification model on a validation set."""
    print("\n" + "=" * 70)
    print("Example 1: Classification Model Evaluation")
    print("=" * 70)

    # Set up reproducibility
    set_seed(42)
    enable_cudnn_determinism(True)

    # Dummy model (replace with Phase 1 model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(
        nn.Linear(224 * 224 * 3, 512),
        nn.ReLU(),
        nn.Linear(512, 10),  # 10 classes
    )

    # Create dummy validation data (batch format from Phase 2 DataLoader)
    def create_dummy_classification_loader(num_batches: int = 5):
        for _ in range(num_batches):
            batch_size = 8
            yield {
                "image": torch.randn(batch_size, 224, 224, 3).to(device),
                "label": torch.randint(0, 10, (batch_size,)).to(device),
            }

    val_loader = create_dummy_classification_loader()

    # Create evaluator
    evaluator = Evaluator(model, device=device)

    # Evaluate
    metrics = evaluator.evaluate_classification(val_loader)

    print(f"Classification Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")


# ============================================================================
# Example 2: Segmentation Model Evaluation
# ============================================================================


def example_segmentation_evaluation() -> None:
    """Evaluate a segmentation model on a validation set."""
    print("\n" + "=" * 70)
    print("Example 2: Segmentation Model Evaluation")
    print("=" * 70)

    # Set up reproducibility
    set_seed(42)
    enable_cudnn_determinism(True)

    # Dummy segmentation model (replace with Phase 1 model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 21, kernel_size=1),  # 21 classes (PASCAL VOC)
    )

    # Create dummy validation data
    def create_dummy_segmentation_loader(num_batches: int = 5):
        for _ in range(num_batches):
            batch_size = 4
            height, width = 128, 128
            yield {
                "image": torch.randn(batch_size, 3, height, width).to(device),
                "mask": torch.randint(0, 21, (batch_size, height, width)).to(device),
            }

    val_loader = create_dummy_segmentation_loader()

    # Create evaluator
    evaluator = Evaluator(model, device=device)

    # Evaluate
    num_classes = 21
    metrics = evaluator.evaluate_segmentation(val_loader, num_classes)

    print(f"Segmentation Metrics:")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"  Mean Dice: {metrics['mean_dice']:.4f}")
    print(f"  Per-class IoU (first 5): {metrics['iou_per_class'][:5]}")
    print(f"  Per-class Dice (first 5): {metrics['dice_per_class'][:5]}")


# ============================================================================
# Example 3: Multi-task Model Evaluation
# ============================================================================


def example_multitask_evaluation() -> None:
    """Evaluate a multi-task model on both classification and segmentation."""
    print("\n" + "=" * 70)
    print("Example 3: Multi-task Model Evaluation")
    print("=" * 70)

    # Set up reproducibility
    set_seed(42)
    enable_cudnn_determinism(True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dummy multi-task model that outputs dict
    class DummyMultiTaskModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 10),  # 10 classes
            )
            self.segmentation_head = nn.Conv2d(64, 21, kernel_size=1)  # 21 classes

        def forward(self, x):
            shared_feat = self.shared(x)
            return {
                "classification": self.classification_head(shared_feat),
                "segmentation": self.segmentation_head(shared_feat),
            }

    model = DummyMultiTaskModel()

    # Create dummy multi-task validation data
    def create_dummy_multitask_loader(num_batches: int = 5):
        for _ in range(num_batches):
            batch_size = 4
            height, width = 128, 128
            yield {
                "image": torch.randn(batch_size, 3, height, width).to(device),
                "label": torch.randint(0, 10, (batch_size,)).to(device),
                "mask": torch.randint(0, 21, (batch_size, height, width)).to(device),
            }

    val_loader = create_dummy_multitask_loader()

    # Create evaluator
    evaluator = Evaluator(model, device=device)

    # Evaluate
    metrics = evaluator.evaluate_multitask(
        val_loader,
        num_classes_classification=10,
        num_classes_segmentation=21,
    )

    print(f"Multi-task Metrics:")
    print(f"  Classification Accuracy: {metrics['classification_accuracy']:.4f}")
    print(f"  Classification Top-5 Accuracy: {metrics['classification_top_5_accuracy']:.4f}")
    print(f"  Segmentation Mean IoU: {metrics['segmentation_mean_iou']:.4f}")
    print(f"  Segmentation Mean Dice: {metrics['segmentation_mean_dice']:.4f}")


# ============================================================================
# Example 4: Reproducibility
# ============================================================================


def example_reproducibility() -> None:
    """Demonstrate reproducible evaluation and training."""
    print("\n" + "=" * 70)
    print("Example 4: Reproducibility")
    print("=" * 70)

    # Run 1: With seed
    print("\nRun 1: With seed 42")
    set_seed(42)
    enable_cudnn_determinism(True)

    random_tensor_1 = torch.randn(3, 3)
    print(f"Random tensor:\n{random_tensor_1}")

    # Run 2: With same seed (should produce identical results)
    print("\nRun 2: With seed 42 again")
    set_seed(42)
    enable_cudnn_determinism(True)

    random_tensor_2 = torch.randn(3, 3)
    print(f"Random tensor:\n{random_tensor_2}")

    # Verify reproducibility
    if torch.allclose(random_tensor_1, random_tensor_2):
        print("\n✓ Reproducibility verified: Both runs produced identical tensors")
    else:
        print("\n✗ Reproducibility failed: Tensors differ")

    # Run 3: With different seed (should differ)
    print("\nRun 3: With seed 123")
    set_seed(123)
    random_tensor_3 = torch.randn(3, 3)
    print(f"Random tensor:\n{random_tensor_3}")

    if not torch.allclose(random_tensor_1, random_tensor_3):
        print("\n✓ Different seeds produce different results as expected")
    else:
        print("\n✗ Different seeds unexpectedly produced same results")


# ============================================================================
# Example 5: Direct Metric Usage
# ============================================================================


def example_direct_metric_usage() -> None:
    """Demonstrate direct usage of metric classes."""
    print("\n" + "=" * 70)
    print("Example 5: Direct Metric Usage")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Classification metrics
    print("\nClassification Metrics:")
    cls_metrics = ClassificationMetrics()

    # Simulate predictions and targets
    for _ in range(5):
        outputs = torch.randn(8, 10).to(device)  # 8 samples, 10 classes
        targets = torch.randint(0, 10, (8,)).to(device)
        cls_metrics.update(outputs, targets)

    cls_results = cls_metrics.compute()
    print(f"  Accuracy: {cls_results['accuracy']:.4f}")
    print(f"  Top-5 Accuracy: {cls_results['top_5_accuracy']:.4f}")

    # Segmentation metrics
    print("\nSegmentation Metrics:")
    seg_metrics = SegmentationMetrics(num_classes=21)

    for _ in range(5):
        outputs = torch.randn(4, 21, 64, 64).to(device)  # 4 samples, 21 classes, 64x64
        targets = torch.randint(0, 21, (4, 64, 64)).to(device)
        seg_metrics.update(outputs, targets)

    seg_results = seg_metrics.compute()
    print(f"  Mean IoU: {seg_results['mean_iou']:.4f}")
    print(f"  Mean Dice: {seg_results['mean_dice']:.4f}")


# ============================================================================
# Example 6: Full Experiment Workflow
# ============================================================================


def example_full_experiment_workflow() -> None:
    """Demonstrate complete experiment workflow with training and evaluation."""
    print("\n" + "=" * 70)
    print("Example 6: Full Experiment Workflow")
    print("=" * 70)

    # Set up reproducibility
    set_seed(42)
    enable_cudnn_determinism(True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dummy classification model
    model = nn.Sequential(
        nn.Linear(32 * 32 * 3, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    # Create dummy dataloaders
    def create_dummy_train_loader(num_batches: int = 3):
        for _ in range(num_batches):
            batch_size = 8
            yield {
                "image": torch.randn(batch_size, 32, 32, 3).to(device),
                "label": torch.randint(0, 10, (batch_size,)).to(device),
            }

    def create_dummy_val_loader(num_batches: int = 3):
        for _ in range(num_batches):
            batch_size = 8
            yield {
                "image": torch.randn(batch_size, 32, 32, 3).to(device),
                "label": torch.randint(0, 10, (batch_size,)).to(device),
            }

    train_loader = create_dummy_train_loader()
    val_loader = create_dummy_val_loader()

    # Create trainer (Phase 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    # Create evaluator (Phase 3)
    evaluator = Evaluator(model, device=device)

    # Create experiment (Phase 3)
    experiment = Experiment(
        model=model,
        trainer=trainer,
        evaluator=evaluator,
        seed=42,
    )

    print("\nStarting experiment workflow...")
    print("(Note: Using dummy data and minimal epochs for demo)")

    # Run experiment (commented out to avoid long demo)
    # results = experiment.run(
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=2,
    #     task="classification",
    # )
    #
    # print(f"\nExperiment Results:")
    # print(f"  Best Epoch: {results['best_epoch']}")
    # print(f"  Best Metrics: {results['best_metrics']}")

    print("✓ Experiment workflow configured successfully")
    print("  (Training skipped for demo purposes)")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    print("\nPhase 3 Integration Examples")
    print("Demonstrating evaluation, metrics, and experiment orchestration")

    # Run examples
    example_direct_metric_usage()
    example_classification_evaluation()
    example_segmentation_evaluation()
    example_multitask_evaluation()
    example_reproducibility()
    example_full_experiment_workflow()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
