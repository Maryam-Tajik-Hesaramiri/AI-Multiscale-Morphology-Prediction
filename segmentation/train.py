#!/usr/bin/env python
"""
train.py
===================

U-Net training with run-balanced and coverage-aware patch sampling.

Purpose
-------
Train the binary SEM segmentation model using the research pipeline's
run-balanced and coverage-aware sampling strategy.

Core configuration
------------------
- U-Net with GroupNorm and bottleneck dropout
- Combined BCE + Dice loss (0.5 / 0.5)
- AdamW optimizer
- Initial learning rate = 1e-3
- Decision threshold = 0.5
- Deterministic training controls
- ReduceLROnPlateau scheduler
- Early stopping based on validation Dice
- Gradient clipping
- Checkpointing and training-history export

Training patches are:
- equally represented across all 9 experimental Runs
- coverage-diverse within each Run

The public portfolio version uses clean module names and avoids storing
machine-specific project paths in exported configuration files.
"""

from __future__ import annotations

import argparse
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
import torch
from torch import Tensor, nn


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATALOADER_FILE = (
    PROJECT_DIR
    / "prepare_dataloader.py"
)

MODEL_FILE = (
    PROJECT_DIR
    / "unet.py"
)

LOSSES_FILE = (
    PROJECT_DIR
    / "losses_metrics.py"
)


# Separate V2 output folder.
OUTPUT_DIR = (
    PROJECT_DIR
    / "training_outputs_10um"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "checkpoints"
)

LOG_DIR = (
    OUTPUT_DIR
    / "logs"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "best_model.pth"
)

LAST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "last_model.pth"
)

HISTORY_CSV_PATH = (
    LOG_DIR
    / "training_history.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "training_summary.txt"
)

CONFIG_PATH = (
    OUTPUT_DIR
    / "training_config.json"
)


# =============================================================================
# 2. TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Hyperparameters kept consistent with the original U-Net."""

    seed: int = 42

    max_epochs: int = 100

    learning_rate: float = 1e-3

    weight_decay: float = 1e-4

    bce_weight: float = 0.5

    dice_weight: float = 0.5

    metric_threshold: float = 0.5

    early_stopping_patience: int = 15

    early_stopping_min_delta: float = 1e-4

    scheduler_patience: int = 5

    scheduler_factor: float = 0.5

    minimum_learning_rate: float = 1e-6

    gradient_clip_norm: float = 1.0

    model_base_channels: int = 32

    model_dropout_probability: float = 0.10

    model_groups: int = 8

    save_every_epoch: bool = True

    def validate(self) -> None:

        if self.max_epochs <= 0:
            raise ValueError(
                "max_epochs must be positive."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay must be non-negative."
            )

        if (
            self.bce_weight < 0
            or self.dice_weight < 0
        ):
            raise ValueError(
                "Loss weights must be non-negative."
            )

        if (
            self.bce_weight
            + self.dice_weight
            <= 0
        ):
            raise ValueError(
                "At least one loss weight "
                "must be positive."
            )

        if not (
            0
            <= self.metric_threshold
            <= 1
        ):
            raise ValueError(
                "metric_threshold must "
                "be between 0 and 1."
            )

        if (
            self.early_stopping_patience
            <= 0
        ):
            raise ValueError(
                "early_stopping_patience "
                "must be positive."
            )

        if self.scheduler_patience < 0:
            raise ValueError(
                "scheduler_patience "
                "must be non-negative."
            )

        if not (
            0
            < self.scheduler_factor
            < 1
        ):
            raise ValueError(
                "scheduler_factor must "
                "be between 0 and 1."
            )


# =============================================================================
# 3. DYNAMIC MODULE IMPORT
# =============================================================================

def load_python_module(
    module_name: str,
    file_path: Path,
):
    """Load a project module from a local Python file."""

    if not file_path.is_file():
        raise FileNotFoundError(
            "Required project script "
            f"was not found:\n{file_path}"
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
            f"Could not load module "
            f"from: {file_path}"
        )

    module = (
        importlib.util.module_from_spec(
            specification
        )
    )

    sys.modules[
        module_name
    ] = module

    specification.loader.exec_module(
        module
    )

    return module


def load_project_modules():

    dataloader_module = (
        load_python_module(
            "prepare_dataloader_module",
            DATALOADER_FILE,
        )
    )

    model_module = (
        load_python_module(
            "unet_module",
            MODEL_FILE,
        )
    )

    losses_module = (
        load_python_module(
            "losses_metrics_module",
            LOSSES_FILE,
        )
    )

    return (
        dataloader_module,
        model_module,
        losses_module,
    )


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

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

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
        CHECKPOINT_DIR,
        LOG_DIR,
        FIGURE_DIR,
    ):

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def save_configuration(
    config: TrainingConfig,
    device: torch.device,
) -> None:

    configuration = asdict(
        config
    )

    configuration.update(
        {
            "experiment":
                "U-Net training with balanced patch sampling",

            "dataloader":
                "prepare_dataloader.py",

            "sampling_strategy":
                (
                    "Equal Run representation "
                    "+ within-Run coverage diversity"
                ),

            "device":
                str(device),

            "python_version":
                platform.python_version(),

            "pytorch_version":
                torch.__version__,

            "cuda_available":
                torch.cuda.is_available(),

            "created_at":
                datetime.now().isoformat(
                    timespec="seconds"
                ),
        }
    )

    with CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            configuration,
            file,
            indent=4,
        )


# =============================================================================
# 6. CHECKPOINTING
# =============================================================================

def build_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_validation_dice: float,
    epochs_without_improvement: int,
    history: List[Dict[str, float]],
    config: TrainingConfig,
) -> Dict[str, Any]:

    return {
        "epoch":
            int(epoch),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_validation_dice":
            float(
                best_validation_dice
            ),

        "epochs_without_improvement":
            int(
                epochs_without_improvement
            ),

        "history":
            history,

        "training_config":
            asdict(config),

        "sampling_strategy":
            "run_balanced_coverage_aware",

        "random_state_python":
            random.getstate(),

        "random_state_numpy":
            np.random.get_state(),

        "random_state_torch":
            torch.get_rng_state(),
    }


def save_checkpoint(
    checkpoint: Dict[str, Any],
    path: Path,
) -> None:

    temporary_path = (
        path.with_suffix(
            path.suffix + ".tmp"
        )
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    temporary_path.replace(
        path
    )


def restore_random_states(
    checkpoint: Dict[str, Any],
) -> None:

    if (
        "random_state_python"
        in checkpoint
    ):

        random.setstate(
            checkpoint[
                "random_state_python"
            ]
        )

    if (
        "random_state_numpy"
        in checkpoint
    ):

        np.random.set_state(
            checkpoint[
                "random_state_numpy"
            ]
        )

    if (
        "random_state_torch"
        in checkpoint
    ):

        torch.set_rng_state(
            checkpoint[
                "random_state_torch"
            ]
        )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> Tuple[
    int,
    float,
    int,
    List[Dict[str, float]],
]:

    if not path.is_file():

        raise FileNotFoundError(
            "Resume checkpoint "
            f"not found:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scheduler.load_state_dict(
        checkpoint[
            "scheduler_state_dict"
        ]
    )

    restore_random_states(
        checkpoint
    )

    next_epoch = (
        int(
            checkpoint["epoch"]
        )
        + 1
    )

    best_validation_dice = float(
        checkpoint.get(
            "best_validation_dice",
            -float("inf"),
        )
    )

    epochs_without_improvement = int(
        checkpoint.get(
            "epochs_without_improvement",
            0,
        )
    )

    history = list(
        checkpoint.get(
            "history",
            [],
        )
    )

    return (
        next_epoch,
        best_validation_dice,
        epochs_without_improvement,
        history,
    )


# =============================================================================
# 7. TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics,
    device: torch.device,
    gradient_clip_norm: float,
) -> Dict[str, float]:

    model.train()

    metrics.reset()

    total_loss = 0.0
    total_bce_loss = 0.0
    total_dice_loss = 0.0
    total_samples = 0

    for batch in loader:

        images: Tensor = (
            batch["image"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
        )

        masks: Tensor = (
            batch["mask"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

        loss, components = (
            criterion(
                logits,
                masks,
                return_components=True,
            )
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Non-finite training "
                "loss encountered."
            )

        loss.backward()

        if gradient_clip_norm > 0:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

        optimizer.step()

        batch_size = (
            images.shape[0]
        )

        total_loss += (
            float(
                loss.item()
            )
            * batch_size
        )

        total_bce_loss += (
            float(
                components[
                    "bce_loss"
                ].item()
            )
            * batch_size
        )

        total_dice_loss += (
            float(
                components[
                    "dice_loss"
                ].item()
            )
            * batch_size
        )

        total_samples += (
            batch_size
        )

        metrics.update(
            logits=logits.detach(),
            targets=masks,
        )

    metric_values = (
        metrics.compute()
    )

    return {
        "loss":
            total_loss
            / total_samples,

        "bce_loss":
            total_bce_loss
            / total_samples,

        "dice_loss":
            total_dice_loss
            / total_samples,

        **metric_values,
    }


# =============================================================================
# 8. VALIDATION
# =============================================================================

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    metrics,
    device: torch.device,
) -> Dict[str, float]:

    model.eval()

    metrics.reset()

    total_loss = 0.0
    total_bce_loss = 0.0
    total_dice_loss = 0.0
    total_samples = 0

    for batch in loader:

        images: Tensor = (
            batch["image"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
        )

        masks: Tensor = (
            batch["mask"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
        )

        logits = model(
            images
        )

        loss, components = (
            criterion(
                logits,
                masks,
                return_components=True,
            )
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Non-finite validation "
                "loss encountered."
            )

        batch_size = (
            images.shape[0]
        )

        total_loss += (
            float(
                loss.item()
            )
            * batch_size
        )

        total_bce_loss += (
            float(
                components[
                    "bce_loss"
                ].item()
            )
            * batch_size
        )

        total_dice_loss += (
            float(
                components[
                    "dice_loss"
                ].item()
            )
            * batch_size
        )

        total_samples += (
            batch_size
        )

        metrics.update(
            logits=logits,
            targets=masks,
        )

    metric_values = (
        metrics.compute()
    )

    return {
        "loss":
            total_loss
            / total_samples,

        "bce_loss":
            total_bce_loss
            / total_samples,

        "dice_loss":
            total_dice_loss
            / total_samples,

        **metric_values,
    }


# =============================================================================
# 9. TRAINING HISTORY
# =============================================================================

HISTORY_COLUMNS = [
    "epoch",
    "learning_rate",
    "epoch_time_seconds",

    "train_loss",
    "train_bce_loss",
    "train_dice_loss",

    "train_dice",
    "train_iou",
    "train_precision",
    "train_recall",
    "train_f1",

    "val_loss",
    "val_bce_loss",
    "val_dice_loss",

    "val_dice",
    "val_iou",
    "val_precision",
    "val_recall",
    "val_f1",
]


def save_history_csv(
    history: List[Dict[str, float]],
) -> None:

    with HISTORY_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=HISTORY_COLUMNS,
        )

        writer.writeheader()

        for row in history:

            writer.writerow(
                {
                    column:
                        row.get(
                            column,
                            "",
                        )
                    for column
                    in HISTORY_COLUMNS
                }
            )


# =============================================================================
# 10. FIGURES
# =============================================================================

def plot_metric_history(
    history,
    train_key,
    validation_key,
    y_label,
    title,
    output_path,
) -> None:

    epochs = [
        row["epoch"]
        for row in history
    ]

    train_values = [
        row[train_key]
        for row in history
    ]

    validation_values = [
        row[validation_key]
        for row in history
    ]

    figure = plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        train_values,
        label="Training",
        linewidth=2,
    )

    plt.plot(
        epochs,
        validation_values,
        label="Validation",
        linewidth=2,
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        y_label
    )

    plt.title(
        title
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_training_figures(
    history,
) -> None:

    if not history:
        return

    plot_metric_history(
        history,
        "train_loss",
        "val_loss",
        "Combined BCE + Dice Loss",
        "U-Net Training and Validation Loss",
        FIGURE_DIR
        / "loss_curve.png",
    )

    plot_metric_history(
        history,
        "train_dice",
        "val_dice",
        "Dice Score",
        "U-Net Training and Validation Dice",
        FIGURE_DIR
        / "dice_curve.png",
    )

    plot_metric_history(
        history,
        "train_iou",
        "val_iou",
        "Intersection over Union",
        "U-Net Training and Validation IoU",
        FIGURE_DIR
        / "iou_curve.png",
    )


# =============================================================================
# 11. SUMMARY
# =============================================================================

def save_summary(
    config,
    device,
    history,
    best_validation_dice,
    best_epoch,
    stopped_early,
    total_training_seconds,
) -> None:

    final_row = (
        history[-1]
        if history
        else {}
    )

    lines = [
        "=" * 78,
        "U-NET TRAINING SUMMARY",
        "=" * 78,

        "Training change:",
        "Run-balanced + coverage-aware patch sampling",

        "",

        f"Completed at:               "
        f"{datetime.now().isoformat(timespec='seconds')}",

        f"Device:                     "
        f"{device}",

        f"Maximum epochs:             "
        f"{config.max_epochs}",

        f"Completed epochs:           "
        f"{len(history)}",

        f"Stopped early:              "
        f"{stopped_early}",

        f"Best epoch:                 "
        f"{best_epoch}",

        f"Best validation Dice:       "
        f"{best_validation_dice:.6f}",

        f"Final validation loss:      "
        f"{final_row.get('val_loss', float('nan')):.6f}",

        f"Final validation IoU:       "
        f"{final_row.get('val_iou', float('nan')):.6f}",

        f"Final validation Precision: "
        f"{final_row.get('val_precision', float('nan')):.6f}",

        f"Final validation Recall:    "
        f"{final_row.get('val_recall', float('nan')):.6f}",

        f"Final validation F1:        "
        f"{final_row.get('val_f1', float('nan')):.6f}",

        f"Total training time:        "
        f"{total_training_seconds / 60:.2f} minutes",

        "",

        f"Best checkpoint:            "
        f"{BEST_CHECKPOINT_PATH.name}",

        f"Last checkpoint:            "
        f"{LAST_CHECKPOINT_PATH.name}",

        "=" * 78,
    ]

    SUMMARY_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# 12. CONSOLE REPORT
# =============================================================================

def print_epoch_report(
    epoch,
    max_epochs,
    learning_rate,
    epoch_seconds,
    train_results,
    validation_results,
    improved,
    epochs_without_improvement,
    patience,
) -> None:

    marker = (
        "  <-- BEST"
        if improved
        else ""
    )

    print("-" * 100)

    print(
        f"Epoch {epoch:03d}/{max_epochs:03d} | "
        f"Time: {epoch_seconds:7.1f}s | "
        f"LR: {learning_rate:.2e}"
        f"{marker}"
    )

    print(
        "Train | "
        f"Loss: {train_results['loss']:.5f} | "
        f"Dice: {train_results['dice']:.5f} | "
        f"IoU: {train_results['iou']:.5f} | "
        f"P: {train_results['precision']:.5f} | "
        f"R: {train_results['recall']:.5f}"
    )

    print(
        "Val   | "
        f"Loss: {validation_results['loss']:.5f} | "
        f"Dice: {validation_results['dice']:.5f} | "
        f"IoU: {validation_results['iou']:.5f} | "
        f"P: {validation_results['precision']:.5f} | "
        f"R: {validation_results['recall']:.5f}"
    )

    print(
        "Early stopping counter: "
        f"{epochs_without_improvement}/"
        f"{patience}"
    )


# =============================================================================
# 13. TRAINING ORCHESTRATION
# =============================================================================

def train(
    config: TrainingConfig,
    resume: bool = False,
) -> None:

    config.validate()

    prepare_output_directories()

    seed_everything(
        config.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    save_configuration(
        config,
        device,
    )

    (
        dataloader_module,
        model_module,
        losses_module,
    ) = load_project_modules()

    (
        train_loader,
        val_loader,
        _,
    ) = dataloader_module.create_dataloaders()

    model = (
        model_module.build_unet(
            in_channels=1,
            out_channels=1,

            base_channels=(
                config.model_base_channels
            ),

            dropout_probability=(
                config.model_dropout_probability
            ),

            groups=(
                config.model_groups
            ),
        ).to(device)
    )

    criterion = (
        losses_module.CombinedBCEDiceLoss(
            bce_weight=(
                config.bce_weight
            ),

            dice_weight=(
                config.dice_weight
            ),
        )
    )

    train_metrics = (
        losses_module.BinarySegmentationMetrics(
            threshold=(
                config.metric_threshold
            ),
            empty_value=1.0,
        )
    )

    validation_metrics = (
        losses_module.BinarySegmentationMetrics(
            threshold=(
                config.metric_threshold
            ),
            empty_value=1.0,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,

            mode="min",

            factor=(
                config.scheduler_factor
            ),

            patience=(
                config.scheduler_patience
            ),

            min_lr=(
                config.minimum_learning_rate
            ),
        )
    )

    start_epoch = 1

    best_validation_dice = (
        -float("inf")
    )

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    if resume:

        (
            start_epoch,
            best_validation_dice,
            epochs_without_improvement,
            history,
        ) = load_checkpoint(
            LAST_CHECKPOINT_PATH,
            model,
            optimizer,
            scheduler,
            device,
        )

        if history:

            best_epoch = int(
                max(
                    history,
                    key=lambda row:
                        row["val_dice"],
                )["epoch"]
            )

        print(
            f"Resuming training "
            f"from epoch {start_epoch}"
        )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print("")
    print("=" * 100)

    print(
        "U-NET TRAINING WITH "
        "RUN-BALANCED SAMPLING"
    )

    print("=" * 100)

    print(
        f"Device:                     "
        f"{device}"
    )

    print(
        f"Trainable parameters:       "
        f"{total_parameters:,}"
    )

    print(
        f"Training batches / epoch:   "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches:         "
        f"{len(val_loader)}"
    )

    print(
        f"Maximum epochs:             "
        f"{config.max_epochs}"
    )

    print(
        f"Initial learning rate:      "
        f"{config.learning_rate:.2e}"
    )

    print(
        "Loss:                       "
        "0.5 BCE + 0.5 Dice"
    )

    print(
        "Sampling:                   "
        "Equal Runs + within-Run coverage diversity"
    )

    print(
        f"Early stopping patience:    "
        f"{config.early_stopping_patience}"
    )

    print(
        "Best-model criterion:       "
        "Validation Dice"
    )

    print(
        f"Output directory:           "
        f"{OUTPUT_DIR.name}"
    )

    print("=" * 100)

    training_start_time = (
        time.perf_counter()
    )

    stopped_early = False

    for epoch in range(
        start_epoch,
        config.max_epochs + 1,
    ):

        epoch_start_time = (
            time.perf_counter()
        )

        # New deterministic coordinates / patch
        # choices every epoch.
        if hasattr(
            train_loader.dataset,
            "set_epoch",
        ):

            train_loader.dataset.set_epoch(
                epoch
            )

        train_results = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            train_metrics,
            device,
            config.gradient_clip_norm,
        )

        validation_results = (
            validate_one_epoch(
                model,
                val_loader,
                criterion,
                validation_metrics,
                device,
            )
        )

        scheduler.step(
            validation_results[
                "loss"
            ]
        )

        current_learning_rate = float(
            optimizer.param_groups[
                0
            ]["lr"]
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start_time
        )

        current_validation_dice = float(
            validation_results[
                "dice"
            ]
        )

        improved = (
            current_validation_dice
            >
            best_validation_dice
            + config.early_stopping_min_delta
        )

        if improved:

            best_validation_dice = (
                current_validation_dice
            )

            best_epoch = epoch

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        history_row = {
            "epoch":
                epoch,

            "learning_rate":
                current_learning_rate,

            "epoch_time_seconds":
                epoch_seconds,

            "train_loss":
                train_results["loss"],

            "train_bce_loss":
                train_results["bce_loss"],

            "train_dice_loss":
                train_results["dice_loss"],

            "train_dice":
                train_results["dice"],

            "train_iou":
                train_results["iou"],

            "train_precision":
                train_results["precision"],

            "train_recall":
                train_results["recall"],

            "train_f1":
                train_results["f1"],

            "val_loss":
                validation_results["loss"],

            "val_bce_loss":
                validation_results["bce_loss"],

            "val_dice_loss":
                validation_results["dice_loss"],

            "val_dice":
                validation_results["dice"],

            "val_iou":
                validation_results["iou"],

            "val_precision":
                validation_results["precision"],

            "val_recall":
                validation_results["recall"],

            "val_f1":
                validation_results["f1"],
        }

        history.append(
            history_row
        )

        checkpoint = build_checkpoint(
            epoch,
            model,
            optimizer,
            scheduler,
            best_validation_dice,
            epochs_without_improvement,
            history,
            config,
        )

        if config.save_every_epoch:

            save_checkpoint(
                checkpoint,
                LAST_CHECKPOINT_PATH,
            )

        if improved:

            save_checkpoint(
                checkpoint,
                BEST_CHECKPOINT_PATH,
            )

        save_history_csv(
            history
        )

        save_training_figures(
            history
        )

        print_epoch_report(
            epoch,
            config.max_epochs,
            current_learning_rate,
            epoch_seconds,
            train_results,
            validation_results,
            improved,
            epochs_without_improvement,
            config.early_stopping_patience,
        )

        if (
            epochs_without_improvement
            >=
            config.early_stopping_patience
        ):

            print("")
            print(
                "Early stopping activated: "
                "validation Dice did not "
                "improve sufficiently."
            )

            stopped_early = True

            break

    total_training_seconds = (
        time.perf_counter()
        - training_start_time
    )

    save_summary(
        config,
        device,
        history,
        best_validation_dice,
        best_epoch,
        stopped_early,
        total_training_seconds,
    )

    print("")
    print("=" * 100)

    print(
        "U-NET TRAINING COMPLETED"
    )

    print("=" * 100)

    print(
        f"Best epoch:                 "
        f"{best_epoch}"
    )

    print(
        f"Best validation Dice:       "
        f"{best_validation_dice:.6f}"
    )

    print(
        f"Total training time:        "
        f"{total_training_seconds / 60:.2f} minutes"
    )

    print(
        f"Best model:                 "
        f"{BEST_CHECKPOINT_PATH.name}"
    )

    print(
        f"History:                    "
        f"{HISTORY_CSV_PATH.name}"
    )

    print("=" * 100)


# =============================================================================
# 14. COMMAND LINE
# =============================================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Train U-Net using "
            "run-balanced coverage-aware sampling."
        )
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from "
            "training_outputs_10um/"
            "checkpoints/last_model.pth"
        ),
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
    )

    return parser.parse_args()


# =============================================================================
# 15. ENTRY POINT
# =============================================================================

def main():

    arguments = (
        parse_arguments()
    )

    config = TrainingConfig(
        max_epochs=(
            arguments.max_epochs
        ),

        learning_rate=(
            arguments.learning_rate
        ),

        early_stopping_patience=(
            arguments
            .early_stopping_patience
        ),
    )

    train(
        config=config,
        resume=arguments.resume,
    )


if __name__ == "__main__":
    main()
