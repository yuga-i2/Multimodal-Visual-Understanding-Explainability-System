"""Phase 3 Extended Example: Statistical Analysis & Result Aggregation.

Demonstrates:
1. Extended metrics (precision, recall, F1, AUC)
2. Bootstrap confidence intervals
3. Result aggregation and CSV export
4. Multi-run comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from models.backbones.cnn import CNNBackbone
from data.datasets import DummyClassificationDataset
from data.transforms import get_train_transforms, get_val_transforms
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from evaluation.confidence_intervals import accuracy_with_ci, f1_with_ci
from evaluation.results_reporting import ResultsAggregator, format_results_table


def example_extended_metrics():
    """Demonstrate extended metrics (precision, recall, F1, AUC)."""
    print("\n" + "=" * 80)
    print("Example 1: Extended Classification Metrics")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create simple model
    model = CNNBackbone(in_channels=3, base_channels=32, num_blocks=1)
    classifier = nn.Sequential(
        model,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),  # 10 classes
    )

    # Create dummy dataset
    val_dataset = DummyClassificationDataset(
        num_samples=100,
        num_classes=10,
        image_size=(128, 128),
        transforms=get_val_transforms(),
    )
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Evaluate with extended metrics
    evaluator = Evaluator(classifier, device=device)
    metrics = evaluator.evaluate_classification(val_loader, num_classes=10)

    # Display results
    table = format_results_table(metrics, task="classification")
    print(table)

    print("\n✓ Extended Metrics Computed:")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Per-class F1 (first 3): {metrics['f1_per_class'][:3]}")
    print(f"  - Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")


def example_confidence_intervals():
    """Demonstrate bootstrap confidence intervals."""
    print("\n" + "=" * 80)
    print("Example 2: Bootstrap Confidence Intervals")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create and evaluate model
    model = CNNBackbone(in_channels=3, base_channels=32, num_blocks=1)
    classifier = nn.Sequential(
        model,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )

    val_dataset = DummyClassificationDataset(
        num_samples=100,
        num_classes=10,
        image_size=(128, 128),
        transforms=get_val_transforms(),
    )
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Get predictions and targets
    all_preds = []
    all_targets = []

    classifier.eval()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = classifier(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute CIs
    print("\nBootstrap Confidence Intervals (95%):")
    print("-" * 60)

    acc_ci = accuracy_with_ci(all_preds, all_targets, num_bootstrap=500)
    print(f"\nAccuracy:")
    print(f"  Point Estimate: {acc_ci['accuracy']:.4f}")
    print(f"  95% CI: [{acc_ci['ci_lower']:.4f}, {acc_ci['ci_upper']:.4f}]")

    f1_ci = f1_with_ci(all_preds, all_targets, num_bootstrap=500, average="macro")
    print(f"\nMacro F1:")
    print(f"  Point Estimate: {f1_ci['f1']:.4f}")
    print(f"  95% CI: [{f1_ci['ci_lower']:.4f}, {f1_ci['ci_upper']:.4f}]")

    print("\n✓ Confidence intervals computed using bootstrap resampling")


def example_multi_run_comparison():
    """Demonstrate result aggregation and comparison across multiple runs."""
    print("\n" + "=" * 80)
    print("Example 3: Multi-Run Aggregation & Comparison")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    aggregator = ResultsAggregator()

    # Simulate 3 runs with different configurations
    configs = [
        {"model": "CNN", "depth": 2, "lr": 0.001},
        {"model": "CNN", "depth": 3, "lr": 0.001},
        {"model": "CNN", "depth": 2, "lr": 0.0005},
    ]

    for run_idx, config in enumerate(configs):
        print(f"\n  Running experiment {run_idx + 1}/{len(configs)}...")

        # Create and train model
        model = CNNBackbone(
            in_channels=3,
            base_channels=32,
            num_blocks=config["depth"],
        )
        classifier = nn.Sequential(
            model,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

        optimizer = optim.Adam(classifier.parameters(), lr=config["lr"])
        loss_fn = nn.CrossEntropyLoss()

        train_dataset = DummyClassificationDataset(
            num_samples=50, num_classes=10, image_size=(128, 128),
            transforms=get_train_transforms()
        )
        val_dataset = DummyClassificationDataset(
            num_samples=20, num_classes=10, image_size=(128, 128),
            transforms=get_val_transforms()
        )

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)

        trainer = Trainer(
            model=classifier,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=True,
            seed=42,
        )

        # Train for 2 epochs
        history = trainer.fit(train_loader, val_loader, epochs=2, patience=None)

        # Evaluate
        evaluator = Evaluator(classifier, device=device)
        train_metrics = {"loss": history["train_loss"][-1]}
        val_metrics = evaluator.evaluate_classification(val_loader, num_classes=10)

        # Add to aggregator
        aggregator.add_run(
            run_id=f"exp_{run_idx:03d}",
            config=config,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

    # Display comparison
    print("\n" + "-" * 80)
    print("RESULTS COMPARISON")
    print("-" * 80)

    df = aggregator.to_dataframe()
    print("\nAll Runs:")
    print(df[["run_id", "config_depth", "config_lr", "val_accuracy", "val_macro_f1"]].to_string(index=False))

    # Best run
    best = aggregator.best_run("accuracy", "val")
    print(f"\n✓ Best Run by Validation Accuracy:")
    print(f"  Run ID: {best['run_id']}")
    print(f"  Accuracy: {best['val_accuracy']:.4f}")
    print(f"  Macro F1: {best['val_macro_f1']:.4f}")

    # Summary stats
    print(f"\n✓ Summary Statistics (Validation Metrics):")
    summary = aggregator.summary_stats("val")
    print(summary.to_string(index=False))

    # Export
    aggregator.to_csv("./results_example_runs.csv")
    aggregator.to_json("./results_example_runs.json")

    print("\n✓ Results exported to CSV and JSON")


def example_results_export():
    """Demonstrate single experiment result export."""
    print("\n" + "=" * 80)
    print("Example 4: Experiment Result Export")
    print("=" * 80)

    from evaluation.results_reporting import export_experiment_results

    # Simulate experiment results
    results = {
        "train_loss": [2.3, 2.1, 1.9, 1.7, 1.5],
        "val_metrics": [
            {"accuracy": 0.40, "macro_f1": 0.35},
            {"accuracy": 0.50, "macro_f1": 0.45},
            {"accuracy": 0.60, "macro_f1": 0.55},
            {"accuracy": 0.65, "macro_f1": 0.62},
            {"accuracy": 0.68, "macro_f1": 0.65},
        ],
        "best_epoch": 4,
        "best_metrics": {"accuracy": 0.68, "macro_f1": 0.65},
    }

    # Export
    export_experiment_results(results, "./results_export", "test_experiment")

    print("\n✓ Experiment results exported to ./results_export/")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PHASE 3 EXTENDED: STATISTICAL ANALYSIS & RESULT AGGREGATION")
    print("=" * 80)

    example_extended_metrics()
    example_confidence_intervals()
    example_multi_run_comparison()
    example_results_export()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE ✓")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Extended metrics (precision, recall, F1, AUC)")
    print("  ✓ Bootstrap confidence intervals")
    print("  ✓ Multi-run aggregation and comparison")
    print("  ✓ Results export (CSV, JSON)")
    print("\nThese features provide statistical rigor and production-ready")
    print("reporting capabilities for ML experiments.\n")
