"""Results aggregation and reporting utilities.

Convert experiment results into pandas DataFrames for easy analysis,
comparison, and export to CSV.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import json
from pathlib import Path


class ResultsAggregator:
    """Aggregate and compare results across multiple runs or experiments."""

    def __init__(self) -> None:
        """Initialize aggregator."""
        self.runs: List[Dict[str, Any]] = []

    def add_run(
        self,
        run_id: str,
        config: Dict[str, Any],
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add a run to the aggregator.

        Args:
            run_id: Unique identifier for the run (e.g., 'exp_001').
            config: Configuration dict (model, dataset, hyperparams, etc.).
            train_metrics: Training metrics dict (loss, accuracy, etc.).
            val_metrics: Validation metrics dict.
            test_metrics: Optional test metrics dict.
        """
        run_data = {
            "run_id": run_id,
            **{f"config_{k}": v for k, v in config.items()},
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        if test_metrics:
            run_data.update({f"test_{k}": v for k, v in test_metrics.items()})

        self.runs.append(run_data)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all runs to a pandas DataFrame.

        Returns:
            DataFrame with one row per run, columns for config and metrics.
        """
        if not self.runs:
            return pd.DataFrame()

        return pd.DataFrame(self.runs)

    def to_csv(self, filepath: str, index: bool = False) -> None:
        """Export results to CSV.

        Args:
            filepath: Path to save CSV file.
            index: Whether to include index column.
        """
        df = self.to_dataframe()
        df.to_csv(filepath, index=index)
        print(f"✓ Results saved to {filepath}")

    def to_json(self, filepath: str, indent: int = 2) -> None:
        """Export results to JSON.

        Args:
            filepath: Path to save JSON file.
            indent: JSON indentation level.
        """
        with open(filepath, "w") as f:
            json.dump(self.runs, f, indent=indent)
        print(f"✓ Results saved to {filepath}")

    def compare_metric(
        self,
        metric_name: str,
        metric_prefix: str = "val",
    ) -> pd.DataFrame:
        """Compare a specific metric across all runs.

        Args:
            metric_name: Name of metric (e.g., 'accuracy').
            metric_prefix: Prefix of the column (e.g., 'val', 'test', 'train').

        Returns:
            DataFrame with run_id and the metric, sorted by metric descending.
        """
        df = self.to_dataframe()
        col_name = f"{metric_prefix}_{metric_name}"

        if col_name not in df.columns:
            raise ValueError(f"Metric column '{col_name}' not found in results.")

        comparison = df[["run_id", col_name]].sort_values(col_name, ascending=False)
        return comparison

    def best_run(self, metric_name: str, metric_prefix: str = "val") -> Dict[str, Any]:
        """Get the best run by a specific metric.

        Args:
            metric_name: Name of metric (e.g., 'accuracy').
            metric_prefix: Prefix of the column.

        Returns:
            Dictionary of the best run's config and metrics.
        """
        df = self.to_dataframe()
        col_name = f"{metric_prefix}_{metric_name}"

        if col_name not in df.columns:
            raise ValueError(f"Metric column '{col_name}' not found in results.")

        best_idx = df[col_name].idxmax()
        return df.iloc[best_idx].to_dict()

    def summary_stats(self, metric_prefix: str = "val") -> pd.DataFrame:
        """Compute summary statistics for all metrics.

        Args:
            metric_prefix: Prefix of columns to summarize.

        Returns:
            DataFrame with mean, std, min, max for each metric.
        """
        df = self.to_dataframe()

        # Get all metric columns (filter by prefix)
        metric_cols = [col for col in df.columns if col.startswith(f"{metric_prefix}_")]

        if not metric_cols:
            raise ValueError(f"No columns found with prefix '{metric_prefix}'.")

        # Compute stats
        stats = pd.DataFrame(
            {
                "metric": metric_cols,
                "mean": [df[col].mean() for col in metric_cols],
                "std": [df[col].std() for col in metric_cols],
                "min": [df[col].min() for col in metric_cols],
                "max": [df[col].max() for col in metric_cols],
            }
        )

        return stats


def format_results_table(metrics_dict: Dict[str, float], task: str = "classification") -> str:
    """Format metrics dict as a readable table string.

    Args:
        metrics_dict: Dictionary of metrics (e.g., from evaluator.compute()).
        task: Task type ('classification' or 'segmentation').

    Returns:
        Formatted table string.
    """
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Metrics ({task.upper()})")
    lines.append(f"{'='*60}")

    # Separate per-class and aggregate metrics
    aggregate = {}
    per_class = {}

    for key, value in metrics_dict.items():
        if isinstance(value, list):
            per_class[key] = value
        else:
            aggregate[key] = value

    # Print aggregate metrics
    if aggregate:
        lines.append("\nAggregate Metrics:")
        for key, value in aggregate.items():
            if isinstance(value, float):
                lines.append(f"  {key:.<40} {value:>10.4f}")
            else:
                lines.append(f"  {key:.<40} {value:>10}")

    # Print per-class metrics (summary)
    if per_class:
        lines.append("\nPer-Class Metrics (first 5 classes):")
        keys = list(per_class.keys())
        max_classes = min(5, len(per_class[keys[0]]) if keys else 0)

        for class_idx in range(max_classes):
            lines.append(f"  Class {class_idx}:")
            for key, values in per_class.items():
                if class_idx < len(values):
                    value = values[class_idx]
                    if isinstance(value, float):
                        lines.append(f"    {key:.<35} {value:>10.4f}")
                    else:
                        lines.append(f"    {key:.<35} {value:>10}")

    lines.append(f"{'='*60}\n")
    return "\n".join(lines)


def export_experiment_results(
    results: Dict[str, Any],
    output_dir: str = "./results",
    experiment_name: str = "experiment",
) -> None:
    """Export complete experiment results to CSV and JSON.

    Args:
        results: Complete experiment results dict (train_loss, val_metrics, etc.).
        output_dir: Directory to save outputs.
        experiment_name: Name prefix for output files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Flatten and prepare data for DataFrame
    data = {
        "epoch": list(range(len(results.get("train_loss", [])))),
        "train_loss": results.get("train_loss", []),
    }

    # Add validation metrics if they exist
    if "val_metrics" in results and results["val_metrics"]:
        first_val_metrics = results["val_metrics"][0]
        for key in first_val_metrics.keys():
            data[f"val_{key}"] = [
                metrics.get(key, None) for metrics in results["val_metrics"]
            ]

    df = pd.DataFrame(data)
    csv_path = output_path / f"{experiment_name}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Experiment results saved to {csv_path}")

    # Save summary as JSON
    summary = {
        "experiment_name": experiment_name,
        "num_epochs": len(results.get("train_loss", [])),
        "best_epoch": results.get("best_epoch", -1),
        "final_train_loss": results.get("train_loss", [None])[-1],
        "best_metrics": results.get("best_metrics", {}),
    }

    json_path = output_path / f"{experiment_name}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to {json_path}")
