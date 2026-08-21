#!/usr/bin/env python
"""
losses_metrics.py
====================

Loss functions and evaluation metrics for binary ZnO semantic segmentation.

This module provides:
    - Binary Cross-Entropy with Logits loss
    - Soft Dice loss
    - Combined BCE + Dice loss
    - Dice coefficient
    - Intersection over Union (IoU)
    - Precision
    - Recall
    - F1-score
    - Confusion-matrix accumulation across batches

Important conventions
---------------------
1. Model outputs must be raw logits with shape:
       (batch_size, 1, height, width)

2. Ground-truth masks must contain float values {0, 1} with the same shape.

3. Sigmoid is NOT applied inside the U-Net.
   This module applies sigmoid only where probabilities or binary predictions
   are required.

4. The default decision threshold is 0.5.

5. Metrics are computed from aggregated pixel counts:
       true positives, false positives, false negatives, true negatives.
   This avoids averaging small batches with unequal statistical weight.

The public portfolio version preserves the loss definitions, metric
conventions, and integrity tests used in the research pipeline.

Usage
-----
Run the standalone tests:

    python losses_metrics.py

Training and evaluation code can import:

    from losses_metrics import CombinedBCEDiceLoss, BinarySegmentationMetrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# =============================================================================
# 1. INPUT VALIDATION
# =============================================================================

def validate_segmentation_tensors(
    logits: Tensor,
    targets: Tensor,
) -> None:
    """Validate the common tensor contract for binary segmentation."""
    if logits.ndim != 4:
        raise ValueError(
            "logits must be a 4D BCHW tensor. "
            f"Received shape: {tuple(logits.shape)}"
        )

    if targets.ndim != 4:
        raise ValueError(
            "targets must be a 4D BCHW tensor. "
            f"Received shape: {tuple(targets.shape)}"
        )

    if logits.shape != targets.shape:
        raise ValueError(
            "logits and targets must have identical shapes. "
            f"Received logits={tuple(logits.shape)}, "
            f"targets={tuple(targets.shape)}"
        )

    if logits.shape[1] != 1:
        raise ValueError(
            "Binary segmentation expects one output channel. "
            f"Received {logits.shape[1]} channels."
        )

    if not torch.is_floating_point(logits):
        raise TypeError("logits must be a floating-point tensor.")

    if not torch.is_floating_point(targets):
        raise TypeError("targets must be a floating-point tensor.")

    if not torch.isfinite(logits).all():
        raise ValueError("logits contain NaN or Inf.")

    if not torch.isfinite(targets).all():
        raise ValueError("targets contain NaN or Inf.")

    if torch.any(targets < 0) or torch.any(targets > 1):
        raise ValueError("targets must be within the interval [0, 1].")


# =============================================================================
# 2. SOFT DICE
# =============================================================================

def soft_dice_score(
    logits: Tensor,
    targets: Tensor,
    smooth: float = 1.0,
    epsilon: float = 1e-7,
) -> Tensor:
    """
    Compute differentiable Dice score from raw logits.

    Dice is computed separately for each image and then averaged across the
    batch. Probabilities remain continuous, so this function is suitable for
    optimization.

    Parameters
    ----------
    logits:
        Raw model output, shape (B, 1, H, W).
    targets:
        Binary ground-truth masks, shape (B, 1, H, W).
    smooth:
        Additive smoothing in numerator and denominator.
    epsilon:
        Numerical stability constant.

    Returns
    -------
    Tensor:
        Scalar mean soft Dice score.
    """
    validate_segmentation_tensors(logits, targets)

    if smooth < 0:
        raise ValueError("smooth must be non-negative.")

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    probabilities = torch.sigmoid(logits)
    targets = targets.to(dtype=probabilities.dtype)

    dimensions = tuple(range(1, probabilities.ndim))

    intersection = torch.sum(
        probabilities * targets,
        dim=dimensions,
    )

    probability_sum = torch.sum(
        probabilities,
        dim=dimensions,
    )

    target_sum = torch.sum(
        targets,
        dim=dimensions,
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        probability_sum + target_sum + smooth + epsilon
    )

    return dice.mean()


class SoftDiceLoss(nn.Module):
    """Differentiable Dice loss: 1 - soft Dice score."""

    def __init__(
        self,
        smooth: float = 1.0,
        epsilon: float = 1e-7,
    ) -> None:
        super().__init__()

        if smooth < 0:
            raise ValueError("smooth must be non-negative.")

        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.smooth = float(smooth)
        self.epsilon = float(epsilon)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        return 1.0 - soft_dice_score(
            logits=logits,
            targets=targets,
            smooth=self.smooth,
            epsilon=self.epsilon,
        )


# =============================================================================
# 3. COMBINED BCE + DICE LOSS
# =============================================================================

class CombinedBCEDiceLoss(nn.Module):
    """
    Weighted combination of BCE-with-logits and Soft Dice loss.

    Total loss:
        total = bce_weight * BCE + dice_weight * DiceLoss

    The default equal weighting is an interpretable and stable starting point.
    Both components are also returned when requested for transparent logging.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        epsilon: float = 1e-7,
        pos_weight: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights must be non-negative.")

        if bce_weight + dice_weight <= 0:
            raise ValueError(
                "At least one loss weight must be greater than zero."
            )

        weight_sum = bce_weight + dice_weight

        # Normalize weights so their sum is exactly one.
        self.bce_weight = float(bce_weight / weight_sum)
        self.dice_weight = float(dice_weight / weight_sum)

        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
        )

        self.dice_loss = SoftDiceLoss(
            smooth=smooth,
            epsilon=epsilon,
        )

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        return_components: bool = False,
    ):
        """
        Compute combined loss.

        Parameters
        ----------
        return_components:
            When False, return only the scalar total loss.
            When True, return:
                total_loss, {"bce_loss": ..., "dice_loss": ...}
        """
        validate_segmentation_tensors(logits, targets)

        targets = targets.to(dtype=logits.dtype)

        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
        )

        if return_components:
            components = {
                "bce_loss": bce.detach(),
                "dice_loss": dice.detach(),
            }
            return total, components

        return total


# =============================================================================
# 4. CONFUSION COUNTS
# =============================================================================

@dataclass
class ConfusionCounts:
    """Pixel-level confusion counts for binary segmentation."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(
        self,
        true_positive: int,
        false_positive: int,
        false_negative: int,
        true_negative: int,
    ) -> None:
        """Add counts from one batch."""
        self.true_positive += int(true_positive)
        self.false_positive += int(false_positive)
        self.false_negative += int(false_negative)
        self.true_negative += int(true_negative)

    def reset(self) -> None:
        """Reset all accumulated counts."""
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0

    @property
    def total_pixels(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )


def confusion_counts_from_logits(
    logits: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
) -> Tuple[int, int, int, int]:
    """
    Convert logits to binary predictions and return TP, FP, FN, TN.

    Counts are detached from the computation graph and calculated using
    Boolean operations.
    """
    validate_segmentation_tensors(logits, targets)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    with torch.no_grad():
        probabilities = torch.sigmoid(logits)
        predictions = probabilities >= threshold
        ground_truth = targets >= 0.5

        true_positive = torch.sum(
            predictions & ground_truth
        ).item()

        false_positive = torch.sum(
            predictions & ~ground_truth
        ).item()

        false_negative = torch.sum(
            ~predictions & ground_truth
        ).item()

        true_negative = torch.sum(
            ~predictions & ~ground_truth
        ).item()

    return (
        int(true_positive),
        int(false_positive),
        int(false_negative),
        int(true_negative),
    )


# =============================================================================
# 5. METRIC CALCULATION
# =============================================================================

def safe_ratio(
    numerator: float,
    denominator: float,
    empty_value: float,
) -> float:
    """
    Return a safe ratio.

    empty_value is used when the denominator is zero.
    """
    if denominator == 0:
        return float(empty_value)

    return float(numerator / denominator)


def metrics_from_confusion_counts(
    counts: ConfusionCounts,
    empty_value: float = 1.0,
) -> Dict[str, float]:
    """
    Calculate segmentation metrics from aggregated pixel counts.

    Empty-case convention
    ---------------------
    When both prediction and target contain no positive pixels:
        Dice, IoU, precision, recall, and F1 are defined as empty_value.

    The default empty_value=1.0 treats a correctly predicted empty mask as a
    perfect result. This convention is stated explicitly for reproducibility.
    """
    tp = float(counts.true_positive)
    fp = float(counts.false_positive)
    fn = float(counts.false_negative)
    tn = float(counts.true_negative)

    dice_denominator = 2.0 * tp + fp + fn
    iou_denominator = tp + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    total = tp + fp + fn + tn

    dice = safe_ratio(
        numerator=2.0 * tp,
        denominator=dice_denominator,
        empty_value=empty_value,
    )

    iou = safe_ratio(
        numerator=tp,
        denominator=iou_denominator,
        empty_value=empty_value,
    )

    precision = safe_ratio(
        numerator=tp,
        denominator=precision_denominator,
        empty_value=empty_value,
    )

    recall = safe_ratio(
        numerator=tp,
        denominator=recall_denominator,
        empty_value=empty_value,
    )

    f1_denominator = precision + recall

    if f1_denominator == 0:
        f1 = float(empty_value if dice_denominator == 0 else 0.0)
    else:
        f1 = 2.0 * precision * recall / f1_denominator

    accuracy = safe_ratio(
        numerator=tp + tn,
        denominator=total,
        empty_value=empty_value,
    )

    specificity = safe_ratio(
        numerator=tn,
        denominator=tn + fp,
        empty_value=empty_value,
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "total_pixels": int(total),
    }


class BinarySegmentationMetrics:
    """
    Accumulate pixel-level metrics over an entire epoch or dataset.

    Recommended use
    ---------------
    metrics = BinarySegmentationMetrics(threshold=0.5)

    for batch in loader:
        logits = model(batch["image"])
        metrics.update(logits, batch["mask"])

    results = metrics.compute()
    """

    def __init__(
        self,
        threshold: float = 0.5,
        empty_value: float = 1.0,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        self.threshold = float(threshold)
        self.empty_value = float(empty_value)
        self.counts = ConfusionCounts()

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self.counts.reset()

    def update(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> None:
        """Accumulate confusion counts from one batch."""
        tp, fp, fn, tn = confusion_counts_from_logits(
            logits=logits,
            targets=targets,
            threshold=self.threshold,
        )

        self.counts.update(
            true_positive=tp,
            false_positive=fp,
            false_negative=fn,
            true_negative=tn,
        )

    def compute(self) -> Dict[str, float]:
        """Return all metrics from the current accumulated counts."""
        return metrics_from_confusion_counts(
            counts=self.counts,
            empty_value=self.empty_value,
        )


# =============================================================================
# 6. CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_batch_metrics(
    logits: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
    empty_value: float = 1.0,
) -> Dict[str, float]:
    """Calculate all hard metrics for a single batch."""
    tp, fp, fn, tn = confusion_counts_from_logits(
        logits=logits,
        targets=targets,
        threshold=threshold,
    )

    counts = ConfusionCounts(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
    )

    return metrics_from_confusion_counts(
        counts=counts,
        empty_value=empty_value,
    )


# =============================================================================
# 7. STANDALONE TESTS
# =============================================================================

def logits_from_binary_prediction(
    binary_prediction: Tensor,
    magnitude: float = 20.0,
) -> Tensor:
    """
    Convert a known binary prediction into confident logits for testing.
    """
    return torch.where(
        binary_prediction > 0.5,
        torch.full_like(binary_prediction, magnitude),
        torch.full_like(binary_prediction, -magnitude),
    )


def run_loss_test() -> None:
    """Verify differentiability and finite combined loss."""
    torch.manual_seed(42)

    logits = torch.randn(
        2,
        1,
        256,
        256,
        requires_grad=True,
    )

    targets = torch.randint(
        low=0,
        high=2,
        size=(2, 1, 256, 256),
    ).float()

    criterion = CombinedBCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    total_loss, components = criterion(
        logits,
        targets,
        return_components=True,
    )

    if not torch.isfinite(total_loss):
        raise RuntimeError("Combined loss is NaN or Inf.")

    total_loss.backward()

    if logits.grad is None:
        raise RuntimeError("Loss backward test did not produce gradients.")

    if not torch.isfinite(logits.grad).all():
        raise RuntimeError("Loss gradients contain NaN or Inf.")

    print("Loss test")
    print("-" * 72)
    print(f"BCE loss:                   {components['bce_loss'].item():.6f}")
    print(f"Soft Dice loss:             {components['dice_loss'].item():.6f}")
    print(f"Combined loss:              {total_loss.item():.6f}")
    print("Backward propagation:       Successful")


def run_metric_tests() -> None:
    """Verify metrics on controlled prediction cases."""
    target = torch.tensor(
        [[[[1, 1],
           [0, 0]]]],
        dtype=torch.float32,
    )

    perfect_prediction = target.clone()

    imperfect_prediction = torch.tensor(
        [[[[1, 0],
           [1, 0]]]],
        dtype=torch.float32,
    )

    empty_target = torch.zeros(
        1,
        1,
        2,
        2,
        dtype=torch.float32,
    )

    empty_prediction = torch.zeros_like(empty_target)

    perfect_metrics = calculate_batch_metrics(
        logits=logits_from_binary_prediction(perfect_prediction),
        targets=target,
    )

    imperfect_metrics = calculate_batch_metrics(
        logits=logits_from_binary_prediction(imperfect_prediction),
        targets=target,
    )

    empty_metrics = calculate_batch_metrics(
        logits=logits_from_binary_prediction(empty_prediction),
        targets=empty_target,
    )

    expected_perfect = ("dice", "iou", "precision", "recall", "f1")

    for metric_name in expected_perfect:
        if abs(perfect_metrics[metric_name] - 1.0) > 1e-8:
            raise RuntimeError(
                f"Perfect-prediction test failed for {metric_name}."
            )

        if abs(empty_metrics[metric_name] - 1.0) > 1e-8:
            raise RuntimeError(
                f"Empty-mask convention failed for {metric_name}."
            )

    # Controlled imperfect case:
    # TP=1, FP=1, FN=1, TN=1
    # Dice/F1=0.5, IoU=1/3, Precision=0.5, Recall=0.5.
    expected_imperfect = {
        "dice": 0.5,
        "iou": 1.0 / 3.0,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }

    for metric_name, expected_value in expected_imperfect.items():
        actual_value = imperfect_metrics[metric_name]

        if abs(actual_value - expected_value) > 1e-8:
            raise RuntimeError(
                f"Metric test failed for {metric_name}: "
                f"expected {expected_value}, received {actual_value}"
            )

    print("\nMetric tests")
    print("-" * 72)
    print(
        "Perfect prediction:        "
        f"Dice={perfect_metrics['dice']:.3f}, "
        f"IoU={perfect_metrics['iou']:.3f}, "
        f"F1={perfect_metrics['f1']:.3f}"
    )
    print(
        "Controlled imperfect:      "
        f"Dice={imperfect_metrics['dice']:.3f}, "
        f"IoU={imperfect_metrics['iou']:.3f}, "
        f"Precision={imperfect_metrics['precision']:.3f}, "
        f"Recall={imperfect_metrics['recall']:.3f}, "
        f"F1={imperfect_metrics['f1']:.3f}"
    )
    print(
        "Correct empty prediction:  "
        f"Dice={empty_metrics['dice']:.3f}, "
        f"IoU={empty_metrics['iou']:.3f}, "
        f"F1={empty_metrics['f1']:.3f}"
    )


def run_accumulator_test() -> None:
    """Verify aggregation across multiple batches."""
    accumulator = BinarySegmentationMetrics(
        threshold=0.5,
        empty_value=1.0,
    )

    targets_batch1 = torch.tensor(
        [[[[1, 0],
           [1, 0]]]],
        dtype=torch.float32,
    )

    predictions_batch1 = torch.tensor(
        [[[[1, 0],
           [0, 0]]]],
        dtype=torch.float32,
    )

    targets_batch2 = torch.tensor(
        [[[[0, 1],
           [0, 1]]]],
        dtype=torch.float32,
    )

    predictions_batch2 = torch.tensor(
        [[[[0, 1],
           [1, 1]]]],
        dtype=torch.float32,
    )

    accumulator.update(
        logits_from_binary_prediction(predictions_batch1),
        targets_batch1,
    )

    accumulator.update(
        logits_from_binary_prediction(predictions_batch2),
        targets_batch2,
    )

    results = accumulator.compute()

    if results["total_pixels"] != 8:
        raise RuntimeError(
            "Metric accumulator total-pixel test failed."
        )

    print("\nAccumulator test")
    print("-" * 72)
    print(f"Total pixels:               {results['total_pixels']}")
    print(f"True positive:              {results['true_positive']}")
    print(f"False positive:             {results['false_positive']}")
    print(f"False negative:             {results['false_negative']}")
    print(f"True negative:              {results['true_negative']}")
    print(f"Aggregated Dice:            {results['dice']:.6f}")
    print(f"Aggregated IoU:             {results['iou']:.6f}")


def main() -> None:
    """Run loss and metric integrity tests."""
    print("=" * 72)
    print("LOSS AND METRIC TESTS")
    print("=" * 72)

    run_loss_test()
    run_metric_tests()
    run_accumulator_test()

    print("=" * 72)
    print("All loss and metric tests completed successfully.")


if __name__ == "__main__":
    main()
