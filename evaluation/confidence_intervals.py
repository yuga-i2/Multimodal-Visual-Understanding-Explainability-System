"""Bootstrap confidence intervals for ML metrics.

Lightweight implementation for computing confidence intervals without
heavy dependencies. Supports accuracy, F1, and IoU metrics.
"""

from typing import Tuple
import numpy as np
import torch


def bootstrap_confidence_interval(
    metric_fn,
    predictions: np.ndarray,
    targets: np.ndarray,
    num_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        metric_fn: Callable that computes metric from (predictions, targets).
        predictions: Model predictions [N, ...].
        targets: Ground truth [N, ...].
        num_bootstrap: Number of bootstrap samples.
        ci: Confidence interval level (0 to 1, default 0.95 for 95% CI).

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound).
    """
    n = len(predictions)
    bootstrap_scores = []

    # Compute point estimate
    point_estimate = metric_fn(predictions, targets)

    # Bootstrap resampling
    np.random.seed(42)  # Deterministic for reproducibility
    for _ in range(num_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_preds = predictions[indices]
        boot_targets = targets[indices]
        boot_score = metric_fn(boot_preds, boot_targets)
        bootstrap_scores.append(boot_score)

    # Compute percentile-based CI
    alpha = 1 - ci
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_scores, lower_percentile)
    upper_bound = np.percentile(bootstrap_scores, upper_percentile)

    return point_estimate, lower_bound, upper_bound


def accuracy_with_ci(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_bootstrap: int = 1000,
    ci: float = 0.95,
) -> dict:
    """Compute accuracy with bootstrap confidence interval.

    Args:
        predictions: Predicted class indices [N].
        targets: Ground truth class indices [N].
        num_bootstrap: Number of bootstrap samples.
        ci: Confidence interval level.

    Returns:
        Dictionary with 'accuracy', 'ci_lower', 'ci_upper', 'ci_level'.
    """
    def _accuracy(preds, targs):
        return (preds == targs).mean()

    point_est, lower, upper = bootstrap_confidence_interval(
        _accuracy, predictions, targets, num_bootstrap, ci
    )

    return {
        "accuracy": point_est,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": ci,
    }


def f1_with_ci(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_bootstrap: int = 1000,
    ci: float = 0.95,
    average: str = "macro",
) -> dict:
    """Compute F1 score with bootstrap confidence interval.

    Args:
        predictions: Predicted class indices [N].
        targets: Ground truth class indices [N].
        num_bootstrap: Number of bootstrap samples.
        ci: Confidence interval level.
        average: 'macro', 'micro', or 'weighted'.

    Returns:
        Dictionary with 'f1', 'ci_lower', 'ci_upper', 'ci_level'.
    """
    def _f1(preds, targs):
        num_classes = len(np.unique(np.concatenate([preds, targs])))
        f1_scores = []

        for class_idx in range(num_classes):
            tp = ((preds == class_idx) & (targs == class_idx)).sum()
            fp = ((preds == class_idx) & (targs != class_idx)).sum()
            fn = ((preds != class_idx) & (targs == class_idx)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            f1_scores.append(f1)

        if average == "macro":
            return np.mean(f1_scores)
        elif average == "micro":
            return (preds == targs).mean()
        elif average == "weighted":
            weights = np.array([(targs == i).sum() for i in range(num_classes)])
            weights = weights / weights.sum()
            return np.average(f1_scores, weights=weights)
        else:
            raise ValueError(f"Unknown average: {average}")

    point_est, lower, upper = bootstrap_confidence_interval(
        _f1, predictions, targets, num_bootstrap, ci
    )

    return {
        "f1": point_est,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": ci,
        "average": average,
    }


def iou_with_ci(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    num_bootstrap: int = 1000,
    ci: float = 0.95,
) -> dict:
    """Compute mean IoU with bootstrap confidence interval.

    Args:
        predictions: Predicted segmentation masks [N, H, W].
        targets: Ground truth segmentation masks [N, H, W].
        num_classes: Number of segmentation classes.
        num_bootstrap: Number of bootstrap samples.
        ci: Confidence interval level.

    Returns:
        Dictionary with 'mean_iou', 'ci_lower', 'ci_upper', 'ci_level'.
    """
    def _mean_iou(preds, targs):
        iou_scores = []

        for class_idx in range(num_classes):
            pred_mask = preds == class_idx
            target_mask = targs == class_idx

            tp = (pred_mask & target_mask).sum()
            fp = (pred_mask & ~target_mask).sum()
            fn = (~pred_mask & target_mask).sum()

            denominator = tp + fp + fn
            iou = tp / denominator if denominator > 0 else 0.0
            iou_scores.append(iou)

        return np.mean(iou_scores)

    point_est, lower, upper = bootstrap_confidence_interval(
        _mean_iou, predictions, targets, num_bootstrap, ci
    )

    return {
        "mean_iou": point_est,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": ci,
    }
