#!/usr/bin/env python
"""
evaluate.py
======================

Independent test evaluation of the trained U-Net.

PURPOSE
-------
Evaluate the U-Net checkpoint trained with:
    Run-balanced + coverage-aware patch sampling.

This evaluation intentionally keeps the main test methodology unchanged:

- Same untouched 9-image Test set
- Same U-Net architecture
- Same SEM preprocessing
- Same patch size = 256 x 256
- Same Test stride = 128 x 128
- Same threshold = 0.5
- Same overlapping-patch reconstruction geometry
- Same full-image metric calculation

NEW DIAGNOSTICS
---------------
Because the original concern was that dense ZnO SEM images sometimes
produced nearly uniform high probability maps, this evaluation adds:

1. 5-panel qualitative figure:
       SEM
       Ground truth
       Probability map
       Binary prediction
       Error map

2. Per-image probability diagnostics:
       Mean probability — all pixels
       Mean probability — true ZnO pixels
       Mean probability — true background pixels
       Probability standard deviation
       Foreground-background probability separation
       Fraction of all pixels with probability >= 0.90
       Fraction of GT foreground pixels with probability >= 0.90
       Fraction of GT background pixels with probability >= 0.90
       Ground-truth coverage
       Predicted coverage
       Coverage bias

These diagnostics help determine whether the model preserves
visible morphology instead of assigning uniformly high ZnO probability.

IMPORTANT
---------
This script does NOT perform:
- training
- regression
- morphology prediction
- LORO
- threshold tuning
- model selection using the Test set

The Test set remains evaluation-only.\n\nThe public portfolio version uses clean module names, portable output metadata, and preserves the research evaluation logic.\n
Expected project structure
--------------------------
10um_binary_segmentation/
├── dataset_split_10um/
│   └── test/
│       ├── images/
│       └── masks/
├── prepare_dataloader.py
├── unet.py
├── losses_metrics.py
├── train.py
├── evaluate.py
└── training_outputs_10um/
    └── checkpoints/
        └── best_model.pth

Generated output
----------------
10um_binary_segmentation/
└── test_evaluation_10um/
    ├── predicted_masks/
    ├── probability_maps/
    ├── error_maps/
    ├── figures/
    ├── test_per_image_metrics.csv
    ├── test_overall_metrics.csv
    ├── test_probability_diagnostics.csv
    ├── test_evaluation_summary.txt
    └── evaluation_config.json

Run
---
python evaluate.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from torch import Tensor
from torch.utils.data import DataLoader


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATALOADER_FILE = PROJECT_DIR / "prepare_dataloader.py"
MODEL_FILE = PROJECT_DIR / "unet.py"

BEST_CHECKPOINT_PATH = (
    PROJECT_DIR
    / "training_outputs_10um"
    / "checkpoints"
    / "best_model.pth"
)

OUTPUT_DIR = PROJECT_DIR / "test_evaluation_10um"

PREDICTED_MASK_DIR = OUTPUT_DIR / "predicted_masks"
PROBABILITY_MAP_DIR = OUTPUT_DIR / "probability_maps"
ERROR_MAP_DIR = OUTPUT_DIR / "error_maps"
FIGURE_DIR = OUTPUT_DIR / "figures"

PER_IMAGE_CSV_PATH = (
    OUTPUT_DIR / "test_per_image_metrics.csv"
)

OVERALL_CSV_PATH = (
    OUTPUT_DIR / "test_overall_metrics.csv"
)

PROBABILITY_DIAGNOSTICS_CSV_PATH = (
    OUTPUT_DIR / "test_probability_diagnostics.csv"
)

SUMMARY_TXT_PATH = (
    OUTPUT_DIR / "test_evaluation_summary.txt"
)

CONFIG_JSON_PATH = (
    OUTPUT_DIR / "evaluation_config.json"
)


# =============================================================================
# 2. EVALUATION CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class EvaluationConfig:

    seed: int = 42

    # Keep identical to training / previous evaluation.
    threshold: float = 0.5

    # Diagnostic only — NOT used to create the binary mask.
    high_probability_threshold: float = 0.90

    expected_test_images: int = 9

    expected_patch_size: Tuple[int, int] = (
        256,
        256,
    )

    expected_test_stride: Tuple[int, int] = (
        128,
        128,
    )

    eval_batch_size: int = 8
    num_workers: int = 0

    save_figures: bool = True

    def validate(self) -> None:

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                "threshold must be in [0, 1]."
            )

        if not 0.0 <= self.high_probability_threshold <= 1.0:
            raise ValueError(
                "high_probability_threshold must be in [0, 1]."
            )

        if self.expected_test_images <= 0:
            raise ValueError(
                "expected_test_images must be positive."
            )

        if self.eval_batch_size <= 0:
            raise ValueError(
                "eval_batch_size must be positive."
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers must be non-negative."
            )


# =============================================================================
# 3. DYNAMIC IMPORT
# =============================================================================

def load_python_module(
    module_name: str,
    file_path: Path,
):
    """Load a local project Python module from a file path."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required project script was not found:\n"
            f"{file_path}"
        )

    specification = (
        importlib.util.spec_from_file_location(
            module_name,
            file_path,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise ImportError(
            f"Could not load module from: {file_path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    sys.modules[module_name] = module

    specification.loader.exec_module(
        module
    )

    return module


# =============================================================================
# 4. REPRODUCIBILITY
# =============================================================================

def seed_everything(
    seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except AttributeError:
        pass


# =============================================================================
# 5. OUTPUT MANAGEMENT
# =============================================================================

def prepare_output_directories() -> None:

    for directory in (
        OUTPUT_DIR,
        PREDICTED_MASK_DIR,
        PROBABILITY_MAP_DIR,
        ERROR_MAP_DIR,
        FIGURE_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# 6. SEGMENTATION METRICS
# =============================================================================

def confusion_counts(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Dict[str, int]:

    prediction = prediction.astype(
        bool,
        copy=False,
    )

    target = target.astype(
        bool,
        copy=False,
    )

    tp = int(
        np.logical_and(
            prediction,
            target,
        ).sum()
    )

    fp = int(
        np.logical_and(
            prediction,
            ~target,
        ).sum()
    )

    fn = int(
        np.logical_and(
            ~prediction,
            target,
        ).sum()
    )

    tn = int(
        np.logical_and(
            ~prediction,
            ~target,
        ).sum()
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> Dict[str, float]:

    dice_den = (
        2 * tp
        + fp
        + fn
    )

    iou_den = (
        tp
        + fp
        + fn
    )

    precision_den = (
        tp
        + fp
    )

    recall_den = (
        tp
        + fn
    )

    dice = (
        1.0
        if dice_den == 0
        else (2.0 * tp) / dice_den
    )

    iou = (
        1.0
        if iou_den == 0
        else tp / iou_den
    )

    if precision_den == 0:

        precision = (
            1.0
            if recall_den == 0
            else 0.0
        )

    else:

        precision = (
            tp / precision_den
        )

    if recall_den == 0:

        recall = (
            1.0
            if precision_den == 0
            else 0.0
        )

    else:

        recall = (
            tp / recall_den
        )

    f1 = (
        1.0
        if dice_den == 0
        else (2.0 * tp) / dice_den
    )

    accuracy = (
        (tp + tn)
        / max(
            tp + fp + fn + tn,
            1,
        )
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


# =============================================================================
# 7. PROBABILITY-MAP DIAGNOSTICS
# =============================================================================

def probability_diagnostics(
    probability_map: np.ndarray,
    target: np.ndarray,
    predicted_mask: np.ndarray,
    high_probability_threshold: float,
) -> Dict[str, float]:
    """
    Quantify whether the probability map is meaningfully structured.

    Especially important for detecting a failure mode in which large portions
    of dense images receive nearly uniform high ZnO probability.
    """

    probability_map = probability_map.astype(
        np.float32,
        copy=False,
    )

    target_bool = target.astype(
        bool,
        copy=False,
    )

    background_bool = ~target_bool

    all_mean = float(
        probability_map.mean()
    )

    all_std = float(
        probability_map.std()
    )

    all_median = float(
        np.median(
            probability_map
        )
    )

    all_p10 = float(
        np.percentile(
            probability_map,
            10,
        )
    )

    all_p90 = float(
        np.percentile(
            probability_map,
            90,
        )
    )

    if target_bool.any():

        foreground_mean = float(
            probability_map[
                target_bool
            ].mean()
        )

        foreground_std = float(
            probability_map[
                target_bool
            ].std()
        )

        foreground_high_fraction = float(
            (
                probability_map[
                    target_bool
                ]
                >= high_probability_threshold
            ).mean()
        )

    else:

        foreground_mean = float("nan")
        foreground_std = float("nan")
        foreground_high_fraction = float("nan")

    if background_bool.any():

        background_mean = float(
            probability_map[
                background_bool
            ].mean()
        )

        background_std = float(
            probability_map[
                background_bool
            ].std()
        )

        background_high_fraction = float(
            (
                probability_map[
                    background_bool
                ]
                >= high_probability_threshold
            ).mean()
        )

    else:

        background_mean = float("nan")
        background_std = float("nan")
        background_high_fraction = float("nan")

    probability_separation = float(
        foreground_mean
        - background_mean
    )

    high_probability_fraction_all = float(
        (
            probability_map
            >= high_probability_threshold
        ).mean()
    )

    ground_truth_coverage = float(
        target.mean()
    )

    predicted_coverage = float(
        predicted_mask.mean()
    )

    coverage_bias = float(
        predicted_coverage
        - ground_truth_coverage
    )

    return {
        "mean_probability_all": all_mean,
        "std_probability_all": all_std,
        "median_probability_all": all_median,
        "p10_probability_all": all_p10,
        "p90_probability_all": all_p90,

        "mean_probability_foreground": foreground_mean,
        "std_probability_foreground": foreground_std,

        "mean_probability_background": background_mean,
        "std_probability_background": background_std,

        "foreground_background_probability_separation":
            probability_separation,

        "fraction_probability_ge_0_90_all":
            high_probability_fraction_all,

        "fraction_probability_ge_0_90_foreground":
            foreground_high_fraction,

        "fraction_probability_ge_0_90_background":
            background_high_fraction,

        "ground_truth_coverage":
            ground_truth_coverage,

        "predicted_coverage":
            predicted_coverage,

        "coverage_bias":
            coverage_bias,
    }


# =============================================================================
# 8. MODEL / CHECKPOINT LOADING
# =============================================================================

def load_best_model(
    model_module,
    device: torch.device,
    threshold: float,
):

    if not BEST_CHECKPOINT_PATH.is_file():

        raise FileNotFoundError(
            "\nbest_model.pth was not found.\n\n"
            f"Expected location:\n"
            f"{BEST_CHECKPOINT_PATH}\n\n"
            "Do NOT substitute last_model.pth."
        )

    checkpoint = torch.load(
        BEST_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    training_config = checkpoint.get(
        "training_config",
        {},
    )

    base_channels = int(
        training_config.get(
            "model_base_channels",
            32,
        )
    )

    dropout_probability = float(
        training_config.get(
            "model_dropout_probability",
            0.10,
        )
    )

    groups = int(
        training_config.get(
            "model_groups",
            8,
        )
    )

    training_threshold = float(
        training_config.get(
            "metric_threshold",
            threshold,
        )
    )

    if (
        abs(
            training_threshold
            - threshold
        )
        > 1e-12
    ):
        raise RuntimeError(
            "Evaluation threshold does not match "
            "checkpoint threshold.\n"
            f"Evaluation = {threshold}\n"
            f"Checkpoint = {training_threshold}"
        )

    model = model_module.build_unet(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        dropout_probability=dropout_probability,
        groups=groups,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    metadata = {

        "checkpoint_epoch":
            int(
                checkpoint.get(
                    "epoch",
                    -1,
                )
            ),

        "best_validation_dice":
            float(
                checkpoint.get(
                    "best_validation_dice",
                    float("nan"),
                )
            ),

        "model_base_channels":
            base_channels,

        "model_dropout_probability":
            dropout_probability,

        "model_groups":
            groups,

        "training_metric_threshold":
            training_threshold,
    }

    return model, metadata


# =============================================================================
# 9. TEST DATASET
# =============================================================================

def build_test_loader(
    dataloader_module,
    config: EvaluationConfig,
) -> Tuple[
    DataLoader,
    Any,
    List[Any],
]:

    # TEST ONLY.
    test_pairs = (
        dataloader_module.discover_pairs(
            "test"
        )
    )

    if (
        len(test_pairs)
        != config.expected_test_images
    ):
        raise RuntimeError(
            f"Expected "
            f"{config.expected_test_images} Test images, "
            f"but found {len(test_pairs)}."
        )

    patch_size = tuple(
        dataloader_module.PATCH_SIZE
    )

    test_stride = tuple(
        dataloader_module.TEST_STRIDE
    )

    if (
        patch_size
        != config.expected_patch_size
    ):
        raise RuntimeError(
            "Patch-size mismatch:\n"
            f"Expected: {config.expected_patch_size}\n"
            f"DataLoader: {patch_size}"
        )

    if (
        test_stride
        != config.expected_test_stride
    ):
        raise RuntimeError(
            "Test-stride mismatch:\n"
            f"Expected: {config.expected_test_stride}\n"
            f"DataLoader: {test_stride}"
        )

    test_dataset = (
        dataloader_module.SlidingWindowDataset(
            pairs=test_pairs,
            patch_size=patch_size,
            stride=test_stride,
        )
    )

    generator = torch.Generator()

    generator.manual_seed(
        config.seed
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=(
            config.eval_batch_size
        ),
        shuffle=False,
        num_workers=(
            config.num_workers
        ),
        pin_memory=(
            torch.cuda.is_available()
        ),
        worker_init_fn=(
            dataloader_module.seed_worker
            if config.num_workers > 0
            else None
        ),
        generator=generator,
        persistent_workers=(
            config.num_workers > 0
        ),
    )

    return (
        test_loader,
        test_dataset,
        test_pairs,
    )


# =============================================================================
# 10. FULL-IMAGE RECONSTRUCTION
# =============================================================================
def create_blending_weight(
    patch_size: Tuple[int, int],
) -> np.ndarray:
    """
    Create a 2D Hann blending window.

    Patch centers receive higher weight than patch edges,
    reducing visible grid / seam artifacts during overlap fusion.

    A small floor prevents zero weights at image boundaries.
    """

    patch_h, patch_w = patch_size

    weight_y = np.hanning(patch_h)
    weight_x = np.hanning(patch_w)

    weight = np.outer(
        weight_y,
        weight_x,
    ).astype(np.float32)

    # Prevent exact zeros at outer image boundaries.
    weight = np.maximum(
        weight,
        1e-3,
    )

    # Normalize max to 1.
    weight /= weight.max()

    return weight


def initialize_reconstruction_buffers(
    test_pairs: List[Any],
    dataloader_module,
) -> Dict[
    str,
    Dict[str, np.ndarray],
]:

    buffers = {}

    for pair in test_pairs:

        image = (
            dataloader_module.to_grayscale(
                dataloader_module.read_image(
                    pair.image_path
                ),
                pair.image_path,
            )
        )

        mask = (
            dataloader_module.prepare_binary_mask(
                dataloader_module.read_image(
                    pair.mask_path
                ),
                pair.mask_path,
            )
        )

        if image.shape != mask.shape:
            raise RuntimeError(
                f"Image/mask shape mismatch "
                f"for {pair.sample_id}: "
                f"{image.shape} vs {mask.shape}"
            )

        height, width = image.shape

        buffers[
            pair.sample_id
        ] = {

            "probability_sum":
                np.zeros(
                    (height, width),
                    dtype=np.float64,
                ),
            "weight_sum":
                np.zeros(
                    (height, width),
                    dtype=np.float64,
                ),    

            "target":
                mask.astype(
                    np.uint8,
                    copy=False,
                ),
        }

    return buffers


@torch.inference_mode()
def reconstruct_test_predictions(
    model,
    test_loader: DataLoader,
    buffers: Dict[
        str,
        Dict[str, np.ndarray],
    ],
    device: torch.device,
    patch_size: Tuple[int, int],
) -> None:

    model.eval()

    blending_weight = create_blending_weight(
        patch_size
    )

    for batch_index, batch in enumerate(
        test_loader,
        start=1,
    ):

        images: Tensor = (
            batch["image"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
        )

        logits = model(
            images
        )

        probabilities = torch.sigmoid(
            logits
        )

        probabilities_np = (
            probabilities[:, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        sample_ids = list(
            batch["sample_id"]
        )

        tops = (
            batch["top"]
            .cpu()
            .numpy()
            .astype(int)
        )

        lefts = (
            batch["left"]
            .cpu()
            .numpy()
            .astype(int)
        )

        original_heights = (
            batch["original_height"]
            .cpu()
            .numpy()
            .astype(int)
        )

        original_widths = (
            batch["original_width"]
            .cpu()
            .numpy()
            .astype(int)
        )

        for i, sample_id in enumerate(
            sample_ids
        ):

            patch_probability = (
                probabilities_np[i]
            )

            top = int(
                tops[i]
            )

            left = int(
                lefts[i]
            )

            original_height = int(
                original_heights[i]
            )

            original_width = int(
                original_widths[i]
            )

            patch_height, patch_width = (
                patch_probability.shape
            )

            bottom = min(
                top + patch_height,
                original_height,
            )

            right = min(
                left + patch_width,
                original_width,
            )

            valid_height = (
                bottom - top
            )

            valid_width = (
                right - left
            )

            if (
                valid_height <= 0
                or valid_width <= 0
            ):
                raise RuntimeError(
                    f"Invalid patch placement "
                    f"for {sample_id}: "
                    f"top={top}, left={left}"
                )

            valid_probability = (
                patch_probability[
                    :valid_height,
                    :valid_width
                ]
            )

            valid_weight = (
                blending_weight[
                    :valid_height,
                    :valid_width
                ]
            )

            buffers[
                sample_id
            ][
                "probability_sum"
            ][
                top:bottom,
                left:right,
            ] += (
                valid_probability
                * valid_weight
            )

            buffers[
                sample_id
            ][
                "weight_sum"
            ][
                top:bottom,
                left:right,
            ] += valid_weight

        if (
            batch_index % 20 == 0
            or batch_index == len(test_loader)
        ):

            processed = min(
                batch_index
                * test_loader.batch_size,
                len(
                    test_loader.dataset
                ),
            )

            print(
                "  Inference patches processed: "
                f"{processed}/"
                f"{len(test_loader.dataset)}"
            )




    

def finalize_probability_map(
    probability_sum: np.ndarray,
    weight_sum: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the full image using weighted overlap blending.
    """

    if np.any(
        weight_sum <= 0
    ):

        missing_pixels = int(
            (
                weight_sum <= 0
            ).sum()
        )

        raise RuntimeError(
            "Weighted reconstruction left "
            f"{missing_pixels} pixels without weight."
        )

    probability_map = (
        probability_sum
        / weight_sum
    ).astype(
        np.float32
    )

    if not np.isfinite(
        probability_map
    ).all():

        raise RuntimeError(
            "Probability map contains NaN or Inf."
        )

    if (
        probability_map.min() < 0.0
        or probability_map.max() > 1.0
    ):

        raise RuntimeError(
            "Probability map contains values "
            "outside [0, 1]."
        )

    return probability_map


# =============================================================================
# 11. ERROR MAP
# =============================================================================

def create_error_map(
    predicted_mask: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Error-map labels:

    0 = True Negative
    1 = True Positive
    2 = False Positive
    3 = False Negative
    """

    prediction = predicted_mask.astype(
        bool
    )

    truth = target.astype(
        bool
    )

    error_map = np.zeros(
        target.shape,
        dtype=np.uint8,
    )

    error_map[
        prediction & truth
    ] = 1

    error_map[
        prediction & (~truth)
    ] = 2

    error_map[
        (~prediction) & truth
    ] = 3

    return error_map


# =============================================================================
# 12. OUTPUT SAVING
# =============================================================================

def save_probability_map(
    sample_id: str,
    probability_map: np.ndarray,
) -> Path:

    path = (
        PROBABILITY_MAP_DIR
        / f"{sample_id}_probability.tif"
    )

    tifffile.imwrite(
        path,
        probability_map.astype(
            np.float32
        ),
        photometric="minisblack",
    )

    return path


def save_predicted_mask(
    sample_id: str,
    predicted_mask: np.ndarray,
) -> Path:

    path = (
        PREDICTED_MASK_DIR
        / f"{sample_id}_pred_mask.tif"
    )

    mask_uint8 = (
        predicted_mask.astype(
            np.uint8,
            copy=False,
        )
        * 255
    )

    tifffile.imwrite(
        path,
        mask_uint8,
        photometric="minisblack",
    )

    return path


def save_error_map(
    sample_id: str,
    error_map: np.ndarray,
) -> Path:

    path = (
        ERROR_MAP_DIR
        / f"{sample_id}_error_map.tif"
    )

    tifffile.imwrite(
        path,
        error_map.astype(
            np.uint8
        ),
        photometric="minisblack",
    )

    return path


def save_comparison_figure(
    sample_id: str,
    image: np.ndarray,
    target: np.ndarray,
    probability_map: np.ndarray,
    predicted_mask: np.ndarray,
    error_map: np.ndarray,
    metrics: Dict[str, float],
    diagnostics: Dict[str, float],
) -> Path:

    path = (
        FIGURE_DIR
        / f"{sample_id}_comparison.png"
    )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(18, 4.5),
        constrained_layout=True,
    )

    # ---------------------------------------------------------
    # SEM
    # ---------------------------------------------------------

    axes[0].imshow(
        image,
        cmap="gray",
    )

    axes[0].set_title(
        "SEM image"
    )

    # ---------------------------------------------------------
    # Ground truth
    # ---------------------------------------------------------

    axes[1].imshow(
        target,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "Ground truth\n"
        f"Coverage="
        f"{diagnostics['ground_truth_coverage'] * 100:.1f}%"
    )

    # ---------------------------------------------------------
    # Probability map
    # ---------------------------------------------------------

    probability_artist = (
        axes[2].imshow(
            probability_map,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
    )

    axes[2].set_title(
        "ZnO probability\n"
        f"BG mean="
        f"{diagnostics['mean_probability_background']:.3f}"
    )

    colorbar = fig.colorbar(
        probability_artist,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Probability"
    )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    axes[3].imshow(
        predicted_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[3].set_title(
        "Prediction\n"
        f"Dice={metrics['dice']:.3f}, "
        f"IoU={metrics['iou']:.3f}"
    )

    # ---------------------------------------------------------
    # Error map
    # ---------------------------------------------------------

    error_artist = (
        axes[4].imshow(
            error_map,
            cmap="tab10",
            vmin=0,
            vmax=3,
        )
    )

    axes[4].set_title(
        "Error map\n"
        "0=TN  1=TP  2=FP  3=FN"
    )

    error_colorbar = fig.colorbar(
        error_artist,
        ax=axes[4],
        fraction=0.046,
        pad=0.04,
        ticks=[0, 1, 2, 3],
    )

    error_colorbar.ax.set_yticklabels(
        [
            "TN",
            "TP",
            "FP",
            "FN",
        ]
    )

    for axis in axes:
        axis.axis(
            "off"
        )

    fig.suptitle(
        (
            f"{sample_id}\n"
            f"P={metrics['precision']:.3f} | "
            f"R={metrics['recall']:.3f} | "
            f"Pred coverage="
            f"{diagnostics['predicted_coverage'] * 100:.1f}% | "
            f"High-P background="
            f"{diagnostics['fraction_probability_ge_0_90_background'] * 100:.1f}%"
        ),
        fontsize=12,
        fontweight="bold",
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path


# =============================================================================
# 13. CSV WRITERS
# =============================================================================

def write_per_image_csv(
    rows: List[
        Dict[str, Any]
    ],
) -> None:

    fieldnames = [
        "sample_id",
        "height",
        "width",

        "ground_truth_coverage",
        "predicted_coverage",
        "coverage_bias",

        "dice",
        "iou",
        "precision",
        "recall",
        "f1",
        "accuracy",

        "tp",
        "fp",
        "fn",
        "tn",

        "predicted_mask_path",
        "probability_map_path",
        "error_map_path",
        "figure_path",
    ]

    with PER_IMAGE_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def write_probability_diagnostics_csv(
    rows: List[
        Dict[str, Any]
    ],
) -> None:

    fieldnames = [
        "sample_id",

        "ground_truth_coverage",
        "predicted_coverage",
        "coverage_bias",

        "mean_probability_all",
        "std_probability_all",
        "median_probability_all",
        "p10_probability_all",
        "p90_probability_all",

        "mean_probability_foreground",
        "std_probability_foreground",

        "mean_probability_background",
        "std_probability_background",

        "foreground_background_probability_separation",

        "fraction_probability_ge_0_90_all",
        "fraction_probability_ge_0_90_foreground",
        "fraction_probability_ge_0_90_background",
    ]

    with (
        PROBABILITY_DIAGNOSTICS_CSV_PATH.open(
            "w",
            newline="",
            encoding="utf-8",
        )
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def write_overall_csv(
    micro_metrics: Dict[str, float],
    macro_metrics: Dict[str, float],
    macro_std: Dict[str, float],
    total_counts: Dict[str, int],
) -> None:

    rows = [

        {
            "aggregation":
                "micro_pixelwise",

            "dice":
                micro_metrics["dice"],

            "iou":
                micro_metrics["iou"],

            "precision":
                micro_metrics["precision"],

            "recall":
                micro_metrics["recall"],

            "f1":
                micro_metrics["f1"],

            "accuracy":
                micro_metrics["accuracy"],

            "tp":
                total_counts["tp"],

            "fp":
                total_counts["fp"],

            "fn":
                total_counts["fn"],

            "tn":
                total_counts["tn"],
        },

        {
            "aggregation":
                "macro_image_mean",

            "dice":
                macro_metrics["dice"],

            "iou":
                macro_metrics["iou"],

            "precision":
                macro_metrics["precision"],

            "recall":
                macro_metrics["recall"],

            "f1":
                macro_metrics["f1"],

            "accuracy":
                macro_metrics["accuracy"],

            "tp": "",
            "fp": "",
            "fn": "",
            "tn": "",
        },

        {
            "aggregation":
                "macro_image_std",

            "dice":
                macro_std["dice"],

            "iou":
                macro_std["iou"],

            "precision":
                macro_std["precision"],

            "recall":
                macro_std["recall"],

            "f1":
                macro_std["f1"],

            "accuracy":
                macro_std["accuracy"],

            "tp": "",
            "fp": "",
            "fn": "",
            "tn": "",
        },
    ]

    fieldnames = [
        "aggregation",
        "dice",
        "iou",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
    ]

    with OVERALL_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# =============================================================================
# 14. CONFIGURATION
# =============================================================================

def write_configuration(
    config: EvaluationConfig,
    device: torch.device,
    checkpoint_metadata: Dict[str, Any],
    number_of_test_patches: int,
) -> None:

    information = asdict(
        config
    )

    information.update(
        {
            "device":
                str(device),

            "python_version":
                platform.python_version(),

            "pytorch_version":
                torch.__version__,

            "numpy_version":
                np.__version__,

            "cuda_available":
                torch.cuda.is_available(),

            "number_of_test_patches":
                number_of_test_patches,

            "created_at":
                datetime.now().isoformat(
                    timespec="seconds"
                ),

            **checkpoint_metadata,
        }
    )

    with CONFIG_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            information,
            file,
            indent=4,
        )


# =============================================================================
# 15. SUMMARY
# =============================================================================

def write_summary(
    config: EvaluationConfig,
    device: torch.device,
    checkpoint_metadata: Dict[str, Any],
    per_image_rows: List[Dict[str, Any]],
    diagnostic_rows: List[Dict[str, Any]],
    micro_metrics: Dict[str, float],
    macro_metrics: Dict[str, float],
    macro_std: Dict[str, float],
    total_counts: Dict[str, int],
    evaluation_seconds: float,
    number_of_test_patches: int,
) -> None:

    bg_probability_values = np.asarray(
        [
            row[
                "mean_probability_background"
            ]
            for row in diagnostic_rows
        ],
        dtype=float,
    )

    high_bg_fraction_values = np.asarray(
        [
            row[
                "fraction_probability_ge_0_90_background"
            ]
            for row in diagnostic_rows
        ],
        dtype=float,
    )

    separation_values = np.asarray(
        [
            row[
                "foreground_background_probability_separation"
            ]
            for row in diagnostic_rows
        ],
        dtype=float,
    )

    lines = [

        "=" * 86,

        "INDEPENDENT U-NET TEST EVALUATION",

        "=" * 86,

        f"Completed at:                  "
        f"{datetime.now().isoformat(timespec='seconds')}",

        f"Device:                        "
        f"{device}",

        f"DataLoader:                    "
        f"{DATALOADER_FILE.name}",

        f"Checkpoint:                    "
        f"{BEST_CHECKPOINT_PATH.name}",

        f"Checkpoint epoch:              "
        f"{checkpoint_metadata['checkpoint_epoch']}",

        f"Best Validation Dice:          "
        f"{checkpoint_metadata['best_validation_dice']:.6f}",

        f"Test images:                   "
        f"{len(per_image_rows)}",

        f"Test patches:                  "
        f"{number_of_test_patches}",

        f"Patch size:                    "
        f"{config.expected_patch_size}",

        f"Test stride:                   "
        f"{config.expected_test_stride}",

        f"Binary threshold:              "
        f"{config.threshold:.3f}",

        f"High-prob diagnostic cutoff:   "
        f"{config.high_probability_threshold:.2f}",

        "Test augmentation:              Disabled",

        "Gradients:                      Disabled",

        "Overlap handling:               "
        "Average probabilities before thresholding",

        "Metric unit:                    "
        "Reconstructed full-size image",

        "",

        "OVERALL TEST METRICS — MICRO / PIXELWISE",

        "-" * 86,

        f"Dice:                          "
        f"{micro_metrics['dice']:.6f}",

        f"IoU:                           "
        f"{micro_metrics['iou']:.6f}",

        f"Precision:                     "
        f"{micro_metrics['precision']:.6f}",

        f"Recall:                        "
        f"{micro_metrics['recall']:.6f}",

        f"F1:                            "
        f"{micro_metrics['f1']:.6f}",

        f"Accuracy:                      "
        f"{micro_metrics['accuracy']:.6f}",

        f"TP:                            "
        f"{total_counts['tp']}",

        f"FP:                            "
        f"{total_counts['fp']}",

        f"FN:                            "
        f"{total_counts['fn']}",

        f"TN:                            "
        f"{total_counts['tn']}",

        "",

        "PER-IMAGE METRICS — MACRO MEAN ± SD",

        "-" * 86,

        f"Dice:                          "
        f"{macro_metrics['dice']:.6f} "
        f"± {macro_std['dice']:.6f}",

        f"IoU:                           "
        f"{macro_metrics['iou']:.6f} "
        f"± {macro_std['iou']:.6f}",

        f"Precision:                     "
        f"{macro_metrics['precision']:.6f} "
        f"± {macro_std['precision']:.6f}",

        f"Recall:                        "
        f"{macro_metrics['recall']:.6f} "
        f"± {macro_std['recall']:.6f}",

        f"F1:                            "
        f"{macro_metrics['f1']:.6f} "
        f"± {macro_std['f1']:.6f}",

        "",

        "PROBABILITY-MAP DIAGNOSTICS — IMAGE MEANS",

        "-" * 86,

        f"Mean GT-background probability:"
        f" {np.nanmean(bg_probability_values):.6f}",

        f"Mean fraction of background "
        f"with P>=0.90:                 "
        f"{np.nanmean(high_bg_fraction_values):.6f}",

        f"Mean FG-BG probability "
        f"separation:                   "
        f"{np.nanmean(separation_values):.6f}",

        "",

        "IMPORTANT INTERPRETATION NOTE",

        "-" * 86,

        "Overall Dice alone does not determine whether the original "
        "dense-image failure mode has been corrected.",

        "The saved probability maps and comparison figures must be "
        "inspected image-by-image, especially for dense ZnO cases.",

        "",

        f"Evaluation time:               "
        f"{evaluation_seconds:.2f} seconds",

        "",

        f"Per-image metrics:             "
        f"{PER_IMAGE_CSV_PATH.name}",

        f"Overall metrics:               "
        f"{OVERALL_CSV_PATH.name}",

        f"Probability diagnostics:       "
        f"{PROBABILITY_DIAGNOSTICS_CSV_PATH.name}",

        f"Predicted masks:               "
        f"{PREDICTED_MASK_DIR.name}",

        f"Probability maps:              "
        f"{PROBABILITY_MAP_DIR.name}",

        f"Error maps:                    "
        f"{ERROR_MAP_DIR.name}",

        f"Figures:                       "
        f"{FIGURE_DIR.name}",

        "=" * 86,
    ]

    SUMMARY_TXT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# 16. EVALUATION
# =============================================================================

def evaluate(
    config: EvaluationConfig,
) -> None:

    config.validate()

    seed_everything(
        config.seed
    )

    prepare_output_directories()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # -------------------------------------------------------------------------
    # Load V4 DataLoader
    # -------------------------------------------------------------------------

    dataloader_module = (
        load_python_module(
            "prepare_dataloader_module",
            DATALOADER_FILE,
        )
    )

    # -------------------------------------------------------------------------
    # Load U-Net architecture
    # -------------------------------------------------------------------------

    model_module = (
        load_python_module(
            "unet_module",
            MODEL_FILE,
        )
    )

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------

    if (
        int(
            dataloader_module.SEED
        )
        != config.seed
    ):
        raise RuntimeError(
            f"Seed mismatch: "
            f"evaluation={config.seed}, "
            f"DataLoader={dataloader_module.SEED}"
        )

    (
        test_loader,
        test_dataset,
        test_pairs,
    ) = build_test_loader(
        dataloader_module,
        config,
    )

    model, checkpoint_metadata = (
        load_best_model(
            model_module=model_module,
            device=device,
            threshold=config.threshold,
        )
    )

    # -------------------------------------------------------------------------
    # Initial report
    # -------------------------------------------------------------------------

    print("")
    print(
        "=" * 86
    )

    print(
        ""
        "INDEPENDENT TEST EVALUATION"
    )

    print(
        "=" * 86
    )

    print(
        f"Device:                    "
        f"{device}"
    )

    print(
        f"DataLoader:                "
        f"{DATALOADER_FILE.name}"
    )

    print(
        f"Checkpoint:                "
        f"{BEST_CHECKPOINT_PATH.name}"
    )

    print(
        f"Checkpoint epoch:          "
        f"{checkpoint_metadata['checkpoint_epoch']}"
    )

    print(
        f"Best Validation Dice:      "
        f"{checkpoint_metadata['best_validation_dice']:.6f}"
    )

    print(
        f"Test images:               "
        f"{len(test_pairs)}"
    )

    print(
        f"Test patches:              "
        f"{len(test_dataset)}"
    )

    print(
        f"Patch size:                "
        f"{dataloader_module.PATCH_SIZE}"
    )

    print(
        f"Test stride:               "
        f"{dataloader_module.TEST_STRIDE}"
    )

    print(
        f"Binary threshold:          "
        f"{config.threshold}"
    )

    print(
        f"High-P diagnostic cutoff:  "
        f"{config.high_probability_threshold}"
    )

    print(
        "Augmentation:              Disabled"
    )

    print(
        "Gradients:                 Disabled"
    )

    print(
        "Overlap fusion:            Hann-weighted probability blending"
    )

    print(
        "=" * 86
    )

    # -------------------------------------------------------------------------
    # Reconstruction
    # -------------------------------------------------------------------------

    buffers = (
        initialize_reconstruction_buffers(
            test_pairs=test_pairs,
            dataloader_module=dataloader_module,
        )
    )

    evaluation_start = (
        time.perf_counter()
    )

    reconstruct_test_predictions(
        model=model,
        test_loader=test_loader,
        buffers=buffers,
        device=device,
        patch_size=tuple(
            dataloader_module.PATCH_SIZE
        ),
    )
    

    # -------------------------------------------------------------------------
    # Per-image evaluation
    # -------------------------------------------------------------------------

    per_image_rows = []

    diagnostic_rows = []

    total_counts = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }

    metric_names = [
        "dice",
        "iou",
        "precision",
        "recall",
        "f1",
        "accuracy",
    ]

    macro_values = {
        name: []
        for name in metric_names
    }

    pair_map = {
        pair.sample_id: pair
        for pair in test_pairs
    }

    print("")
    print(
        "Per-image results"
    )
    print(
        "-" * 120
    )

    for sample_id in sorted(
        buffers
    ):

        buffer = (
            buffers[sample_id]
        )

        probability_map = (
            finalize_probability_map(
            buffer[
                "probability_sum"
            ],
            buffer[
                "weight_sum"
            ],
        )
    )


        predicted_mask = (
            probability_map
            >= config.threshold
        ).astype(
            np.uint8
        )

        target = (
            buffer[
                "target"
            ].astype(
                np.uint8,
                copy=False,
            )
        )

        counts = (
            confusion_counts(
                prediction=predicted_mask,
                target=target,
            )
        )

        image_metrics = (
            metrics_from_counts(
                **counts
            )
        )

        diagnostics = (
            probability_diagnostics(
                probability_map=probability_map,
                target=target,
                predicted_mask=predicted_mask,
                high_probability_threshold=(
                    config.high_probability_threshold
                ),
            )
        )

        error_map = (
            create_error_map(
                predicted_mask=predicted_mask,
                target=target,
            )
        )

        # -----------------------------------------------------
        # Aggregation
        # -----------------------------------------------------

        for key in total_counts:

            total_counts[
                key
            ] += counts[
                key
            ]

        for name in metric_names:

            macro_values[
                name
            ].append(
                image_metrics[
                    name
                ]
            )

        # -----------------------------------------------------
        # Save arrays
        # -----------------------------------------------------

        mask_path = (
            save_predicted_mask(
                sample_id,
                predicted_mask,
            )
        )

        probability_path = (
            save_probability_map(
                sample_id,
                probability_map,
            )
        )

        error_path = (
            save_error_map(
                sample_id,
                error_map,
            )
        )

        pair = (
            pair_map[
                sample_id
            ]
        )

        image = (
            dataloader_module.to_grayscale(
                dataloader_module.read_image(
                    pair.image_path
                ),
                pair.image_path,
            )
        )

        figure_path = ""

        if config.save_figures:

            figure_path = str(
                save_comparison_figure(
                    sample_id=sample_id,
                    image=image,
                    target=target,
                    probability_map=probability_map,
                    predicted_mask=predicted_mask,
                    error_map=error_map,
                    metrics=image_metrics,
                    diagnostics=diagnostics,
                )
            )

        # -----------------------------------------------------
        # Main metrics row
        # -----------------------------------------------------

        row = {

            "sample_id":
                sample_id,

            "height":
                int(
                    target.shape[0]
                ),

            "width":
                int(
                    target.shape[1]
                ),

            "ground_truth_coverage":
                diagnostics[
                    "ground_truth_coverage"
                ],

            "predicted_coverage":
                diagnostics[
                    "predicted_coverage"
                ],

            "coverage_bias":
                diagnostics[
                    "coverage_bias"
                ],

            "dice":
                image_metrics[
                    "dice"
                ],

            "iou":
                image_metrics[
                    "iou"
                ],

            "precision":
                image_metrics[
                    "precision"
                ],

            "recall":
                image_metrics[
                    "recall"
                ],

            "f1":
                image_metrics[
                    "f1"
                ],

            "accuracy":
                image_metrics[
                    "accuracy"
                ],

            "tp":
                counts["tp"],

            "fp":
                counts["fp"],

            "fn":
                counts["fn"],

            "tn":
                counts["tn"],

            "predicted_mask_path":
                mask_path.name,

            "probability_map_path":
                probability_path.name,

            "error_map_path":
                error_path.name,

            "figure_path":
                Path(figure_path).name if figure_path else "",
        }

        per_image_rows.append(
            row
        )

        diagnostic_row = {
            "sample_id":
                sample_id,
            **diagnostics,
        }

        diagnostic_rows.append(
            diagnostic_row
        )

        print(
            f"{sample_id:<30} "
            f"Dice={image_metrics['dice']:.4f} | "
            f"IoU={image_metrics['iou']:.4f} | "
            f"P={image_metrics['precision']:.4f} | "
            f"R={image_metrics['recall']:.4f} | "
            f"GTcov={diagnostics['ground_truth_coverage'] * 100:6.2f}% | "
            f"PredCov={diagnostics['predicted_coverage'] * 100:6.2f}% | "
            f"BGmeanP={diagnostics['mean_probability_background']:.4f} | "
            f"BG>=0.9="
            f"{diagnostics['fraction_probability_ge_0_90_background'] * 100:6.2f}%"
        )

    # -------------------------------------------------------------------------
    # Overall metrics
    # -------------------------------------------------------------------------

    evaluation_seconds = (
        time.perf_counter()
        - evaluation_start
    )

    micro_metrics = (
        metrics_from_counts(
            **total_counts
        )
    )

    macro_metrics = {
        name: float(
            np.mean(
                macro_values[
                    name
                ]
            )
        )
        for name in metric_names
    }

    macro_std = {
        name: float(
            np.std(
                macro_values[
                    name
                ],
                ddof=(
                    1
                    if len(
                        macro_values[
                            name
                        ]
                    ) > 1
                    else 0
                ),
            )
        )
        for name in metric_names
    }

    # -------------------------------------------------------------------------
    # Save tables / config / summary
    # -------------------------------------------------------------------------

    write_per_image_csv(
        per_image_rows
    )

    write_probability_diagnostics_csv(
        diagnostic_rows
    )

    write_overall_csv(
        micro_metrics=micro_metrics,
        macro_metrics=macro_metrics,
        macro_std=macro_std,
        total_counts=total_counts,
    )

    write_configuration(
        config=config,
        device=device,
        checkpoint_metadata=checkpoint_metadata,
        number_of_test_patches=(
            len(test_dataset)
        ),
    )

    write_summary(
        config=config,
        device=device,
        checkpoint_metadata=checkpoint_metadata,
        per_image_rows=per_image_rows,
        diagnostic_rows=diagnostic_rows,
        micro_metrics=micro_metrics,
        macro_metrics=macro_metrics,
        macro_std=macro_std,
        total_counts=total_counts,
        evaluation_seconds=evaluation_seconds,
        number_of_test_patches=(
            len(test_dataset)
        ),
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print("")
    print(
        "=" * 86
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=" * 86
    )

    print(
        "Micro / pixelwise aggregation"
    )

    print(
        f"Dice:       "
        f"{micro_metrics['dice']:.6f}"
    )

    print(
        f"IoU:        "
        f"{micro_metrics['iou']:.6f}"
    )

    print(
        f"Precision:  "
        f"{micro_metrics['precision']:.6f}"
    )

    print(
        f"Recall:     "
        f"{micro_metrics['recall']:.6f}"
    )

    print(
        f"F1:         "
        f"{micro_metrics['f1']:.6f}"
    )

    print("")

    print(
        "Macro / per-image mean ± SD"
    )

    print(
        f"Dice:       "
        f"{macro_metrics['dice']:.6f} "
        f"± {macro_std['dice']:.6f}"
    )

    print(
        f"IoU:        "
        f"{macro_metrics['iou']:.6f} "
        f"± {macro_std['iou']:.6f}"
    )

    print(
        f"Precision:  "
        f"{macro_metrics['precision']:.6f} "
        f"± {macro_std['precision']:.6f}"
    )

    print(
        f"Recall:     "
        f"{macro_metrics['recall']:.6f} "
        f"± {macro_std['recall']:.6f}"
    )

    print("")

    print(
        "IMPORTANT:"
    )

    print(
        "Do not approve or reject from Dice alone."
    )

    print(
        "Inspect all 9 probability maps, especially dense ZnO images."
    )

    print("")

    print(
        f"Per-image metrics:       "
        f"{PER_IMAGE_CSV_PATH.name}"
    )

    print(
        f"Probability diagnostics: "
        f"{PROBABILITY_DIAGNOSTICS_CSV_PATH.name}"
    )

    print(
        f"Summary:                 "
        f"{SUMMARY_TXT_PATH.name}"
    )

    print(
        f"Figures:                 "
        f"{FIGURE_DIR.name}"
    )

    print(
        "=" * 86
    )


# =============================================================================
# 17. ENTRY POINT
# =============================================================================

def main() -> None:

    config = EvaluationConfig()

    evaluate(
        config
    )


if __name__ == "__main__":
    main()
