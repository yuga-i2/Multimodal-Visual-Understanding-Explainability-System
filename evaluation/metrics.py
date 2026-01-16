"""Evaluation metrics for classification and segmentation tasks.

Pure PyTorch implementations of standard metrics:
- Classification: Accuracy, Top-k Accuracy, Precision, Recall, F1, ROC-AUC
- Segmentation: Dice coefficient, IoU (Jaccard index), Precision, Recall per-class
"""

import torch
import numpy as np


class ClassificationMetrics:
    """Accumulates and computes classification metrics.
    
    Supports:
    - Accuracy, Top-k Accuracy
    - Per-class Precision, Recall, F1
    - Macro & Micro F1
    - ROC-AUC (binary and multi-class)
    """

    def __init__(self, num_classes: int = None) -> None:
        """Initialize metric accumulators.
        
        Args:
            num_classes: Number of classes. Required for per-class metrics and AUC.
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators."""
        self.total_correct = 0
        self.total_top_k_correct = 0
        self.total_samples = 0
        
        # Per-class confusion matrix components
        if self.num_classes is not None:
            self.tp = torch.zeros(self.num_classes)  # True positives per class
            self.fp = torch.zeros(self.num_classes)  # False positives per class
            self.fn = torch.zeros(self.num_classes)  # False negatives per class
        
        # For AUC computation
        self.all_predictions = []
        self.all_targets = []

    def update(self, outputs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> None:
        """Update metrics with batch predictions.

        Args:
            outputs: Model outputs (logits) [B, C] where C is number of classes.
            targets: Ground truth labels [B] (long tensor).
            k: For top-k accuracy (default 5).

        Raises:
            ValueError: If outputs and targets have incompatible shapes.
        """
        if outputs.size(0) != targets.size(0):
            raise ValueError(
                f"Batch size mismatch: outputs {outputs.size(0)} vs targets {targets.size(0)}"
            )

        batch_size = outputs.size(0)
        num_classes = outputs.size(1)

        # Accuracy
        predictions = outputs.argmax(dim=1)
        correct = (predictions == targets).sum().item()
        self.total_correct += correct

        # Top-k accuracy
        k_actual = min(k, num_classes)
        _, top_k_preds = outputs.topk(k_actual, dim=1)
        top_k_correct = top_k_preds.eq(targets.unsqueeze(1).expand_as(top_k_preds)).any(dim=1).sum().item()
        self.total_top_k_correct += top_k_correct

        self.total_samples += batch_size
        
        # Per-class TP/FP/FN for Precision, Recall, F1
        if self.num_classes is not None:
            predictions = predictions.cpu()
            targets = targets.cpu()
            for class_idx in range(self.num_classes):
                pred_mask = predictions == class_idx
                target_mask = targets == class_idx
                
                tp = (pred_mask & target_mask).sum().item()
                fp = (pred_mask & ~target_mask).sum().item()
                fn = (~pred_mask & target_mask).sum().item()
                
                self.tp[class_idx] += tp
                self.fp[class_idx] += fp
                self.fn[class_idx] += fn
        
        # Store predictions/targets for AUC
        self.all_predictions.append(outputs.detach().cpu())
        self.all_targets.append(targets.cpu())

    def compute(self) -> dict:
        """Compute accumulated metrics.

        Returns:
            Dictionary with keys:
            - 'accuracy': Classification accuracy (0 to 1)
            - 'top_5_accuracy': Top-5 accuracy (0 to 1)
            - 'precision_per_class': Per-class precision (if num_classes set)
            - 'recall_per_class': Per-class recall (if num_classes set)
            - 'f1_per_class': Per-class F1 (if num_classes set)
            - 'macro_f1': Macro-averaged F1 (if num_classes set)
            - 'micro_f1': Micro-averaged F1 (if num_classes set)
            - 'roc_auc': ROC-AUC score (binary: single value; multi-class: macro average)
        """
        results = {}
        
        if self.total_samples == 0:
            results["accuracy"] = 0.0
            results["top_5_accuracy"] = 0.0
            if self.num_classes is not None:
                results["precision_per_class"] = [0.0] * self.num_classes
                results["recall_per_class"] = [0.0] * self.num_classes
                results["f1_per_class"] = [0.0] * self.num_classes
                results["macro_f1"] = 0.0
                results["micro_f1"] = 0.0
                results["roc_auc"] = 0.0
            return results
        
        results["accuracy"] = self.total_correct / self.total_samples
        results["top_5_accuracy"] = self.total_top_k_correct / self.total_samples
        
        # Compute per-class metrics if num_classes is set
        if self.num_classes is not None:
            precision_per_class = []
            recall_per_class = []
            f1_per_class = []
            
            for class_idx in range(self.num_classes):
                # Precision = TP / (TP + FP)
                denom_p = self.tp[class_idx] + self.fp[class_idx]
                precision = self.tp[class_idx] / denom_p if denom_p > 0 else 0.0
                precision_per_class.append(precision.item() if isinstance(precision, torch.Tensor) else precision)
                
                # Recall = TP / (TP + FN)
                denom_r = self.tp[class_idx] + self.fn[class_idx]
                recall = self.tp[class_idx] / denom_r if denom_r > 0 else 0.0
                recall_per_class.append(recall.item() if isinstance(recall, torch.Tensor) else recall)
                
                # F1 = 2 * (Precision * Recall) / (Precision + Recall)
                if precision_per_class[-1] + recall_per_class[-1] > 0:
                    f1 = 2 * (precision_per_class[-1] * recall_per_class[-1]) / (precision_per_class[-1] + recall_per_class[-1])
                else:
                    f1 = 0.0
                f1_per_class.append(f1)
            
            results["precision_per_class"] = precision_per_class
            results["recall_per_class"] = recall_per_class
            results["f1_per_class"] = f1_per_class
            results["macro_f1"] = np.mean(f1_per_class)
            
            # Micro F1 = accuracy (for multi-class)
            results["micro_f1"] = results["accuracy"]
            
            # ROC-AUC computation
            results["roc_auc"] = self._compute_roc_auc()
        
        return results
    
    def _compute_roc_auc(self) -> float:
        """Compute ROC-AUC score.
        
        For binary classification: single AUC value
        For multi-class: macro-averaged AUC (one-vs-rest)
        
        Returns:
            AUC score (0 to 1)
        """
        if not self.all_predictions:
            return 0.0
        
        # Concatenate all batches
        all_preds = torch.cat(self.all_predictions, dim=0)  # [N, C]
        all_targets = torch.cat(self.all_targets, dim=0)    # [N]
        
        if self.num_classes == 2:
            # Binary classification: AUC for positive class
            probs = torch.softmax(all_preds, dim=1)[:, 1].numpy()
            targets = all_targets.numpy()
            return self._roc_auc_score(targets, probs)
        elif self.num_classes > 2:
            # Multi-class: macro average of one-vs-rest AUCs
            probs = torch.softmax(all_preds, dim=1).numpy()
            targets = all_targets.numpy()
            auc_scores = []
            
            for class_idx in range(self.num_classes):
                binary_targets = (targets == class_idx).astype(int)
                class_probs = probs[:, class_idx]
                auc = self._roc_auc_score(binary_targets, class_probs)
                auc_scores.append(auc)
            
            return np.mean(auc_scores)
        else:
            return 0.0
    
    @staticmethod
    def _roc_auc_score(targets: np.ndarray, predictions: np.ndarray) -> float:
        """Compute AUC via trapezoidal rule (no sklearn dependency).
        
        Args:
            targets: Binary labels [N]
            predictions: Predicted probabilities [N]
        
        Returns:
            AUC score (0 to 1)
        """
        # Sort by predictions (descending)
        sorted_indices = np.argsort(-predictions)
        sorted_targets = targets[sorted_indices]
        
        # Number of positives and negatives
        n_pos = np.sum(sorted_targets)
        n_neg = len(sorted_targets) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0.0
        
        # Compute TPR and FPR at each threshold
        tpr = np.cumsum(sorted_targets) / n_pos
        fpr = np.cumsum(1 - sorted_targets) / n_neg
        
        # Add (0, 0) and (1, 1) for proper ROC curve
        tpr = np.insert(tpr, 0, 0)
        fpr = np.insert(fpr, 0, 0)
        
        # Compute AUC via trapezoidal rule
        auc = np.trapz(tpr, fpr)
        return float(auc)


class SegmentationMetrics:
    """Accumulates and computes segmentation metrics.
    
    Supports:
    - Per-class and mean IoU, Dice
    - Per-class Precision and Recall
    """

    def __init__(self, num_classes: int) -> None:
        """Initialize metric accumulators.

        Args:
            num_classes: Number of segmentation classes.
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators."""
        # Per-class IoU: track TP, FP, FN for each class
        self.iou_tp = torch.zeros(self.num_classes)
        self.iou_fp = torch.zeros(self.num_classes)
        self.iou_fn = torch.zeros(self.num_classes)

        # Dice coefficient per class
        self.dice_tp = torch.zeros(self.num_classes)
        self.dice_total_pred = torch.zeros(self.num_classes)
        self.dice_total_target = torch.zeros(self.num_classes)

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics with batch predictions.

        Args:
            outputs: Model outputs (logits) [B, C, H, W].
            targets: Ground truth masks [B, H, W] (long tensor with class indices).

        Raises:
            ValueError: If outputs and targets have incompatible shapes.
        """
        if outputs.size(0) != targets.size(0):
            raise ValueError(
                f"Batch size mismatch: outputs {outputs.size(0)} vs targets {targets.size(0)}"
            )
        if outputs.size(2) != targets.size(1) or outputs.size(3) != targets.size(2):
            raise ValueError(
                f"Spatial size mismatch: outputs {outputs.shape} vs targets {targets.shape}"
            )

        # Get predictions (argmax over class dimension)
        predictions = outputs.argmax(dim=1)  # [B, H, W]

        # Move to CPU for accumulation
        predictions = predictions.cpu()
        targets = targets.cpu()

        # Compute per-class metrics
        for class_idx in range(self.num_classes):
            pred_mask = predictions == class_idx
            target_mask = targets == class_idx

            # IoU components
            tp = (pred_mask & target_mask).sum().item()
            fp = (pred_mask & ~target_mask).sum().item()
            fn = (~pred_mask & target_mask).sum().item()

            self.iou_tp[class_idx] += tp
            self.iou_fp[class_idx] += fp
            self.iou_fn[class_idx] += fn

            # Dice components
            self.dice_tp[class_idx] += tp
            self.dice_total_pred[class_idx] += pred_mask.sum().item()
            self.dice_total_target[class_idx] += target_mask.sum().item()

    def compute(self) -> dict:
        """Compute accumulated metrics.

        Returns:
            Dictionary with keys:
            - 'iou_per_class': Per-class IoU values [num_classes]
            - 'mean_iou': Mean IoU across classes
            - 'dice_per_class': Per-class Dice coefficients [num_classes]
            - 'mean_dice': Mean Dice coefficient across classes
            - 'precision_per_class': Per-class precision
            - 'recall_per_class': Per-class recall
        """
        # Compute per-class IoU
        iou_per_class = torch.zeros(self.num_classes)
        precision_per_class = torch.zeros(self.num_classes)
        recall_per_class = torch.zeros(self.num_classes)
        
        for class_idx in range(self.num_classes):
            # IoU
            denominator = self.iou_tp[class_idx] + self.iou_fp[class_idx] + self.iou_fn[class_idx]
            if denominator > 0:
                iou_per_class[class_idx] = self.iou_tp[class_idx] / denominator
            else:
                iou_per_class[class_idx] = 0.0
            
            # Precision = TP / (TP + FP)
            denom_p = self.iou_tp[class_idx] + self.iou_fp[class_idx]
            if denom_p > 0:
                precision_per_class[class_idx] = self.iou_tp[class_idx] / denom_p
            else:
                precision_per_class[class_idx] = 0.0
            
            # Recall = TP / (TP + FN)
            denom_r = self.iou_tp[class_idx] + self.iou_fn[class_idx]
            if denom_r > 0:
                recall_per_class[class_idx] = self.iou_tp[class_idx] / denom_r
            else:
                recall_per_class[class_idx] = 0.0

        # Compute per-class Dice
        dice_per_class = torch.zeros(self.num_classes)
        for class_idx in range(self.num_classes):
            denominator = self.dice_total_pred[class_idx] + self.dice_total_target[class_idx]
            if denominator > 0:
                dice_per_class[class_idx] = 2.0 * self.dice_tp[class_idx] / denominator
            else:
                dice_per_class[class_idx] = 0.0

        return {
            "iou_per_class": iou_per_class.tolist(),
            "mean_iou": iou_per_class.mean().item(),
            "dice_per_class": dice_per_class.tolist(),
            "mean_dice": dice_per_class.mean().item(),
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
        }
