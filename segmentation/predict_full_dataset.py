#!/usr/bin/env python
"""
predict_full_dataset.py
=======================

Full-dataset inference for 10 µm ZnO SEM images.

Applies the trained U-Net to the complete SEM dataset using overlapping
sliding-window inference and reconstructs full-resolution probability maps
and binary segmentation masks.

Outputs
-------
- Full-resolution binary masks
- Full-resolution probability maps
- Per-image ZnO coverage
- Inference manifest
- Configuration metadata
- Run-level coverage summary

Notes
-----
Overlapping patch probabilities are averaged before applying the binary
threshold. Source SEM images are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import tifffile
import torch

from prepare_dataloader import (
    PATCH_SIZE,
    TEST_STRIDE,
    SEED,
    SUPPORTED_EXTENSIONS,
    read_image,
    to_grayscale,
    normalize_sem_image,
)
from unet import build_unet


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CHECKPOINT = (
    SCRIPT_DIR
    / "training_outputs"
    / "checkpoints"
    / "best_model.pth"
)

DEFAULT_INPUT_DIR = (
    SCRIPT_DIR.parent
    / "ZnO_SEM_dataset_5um"
)

DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR
    / "full_dataset_predictions"
)


@dataclass(frozen=True)
class InferenceConfig:
    """Inference settings."""

    seed: int = SEED
    patch_size: Tuple[int, int] = tuple(PATCH_SIZE)
    stride: Tuple[int, int] = tuple(TEST_STRIDE)

    batch_size: int = 8
    threshold: float = 0.5

    save_probability_maps: bool = True
    save_binary_masks: bool = True

    def validate(self) -> None:
        patch_h, patch_w = self.patch_size
        stride_h, stride_w = self.stride

        if patch_h <= 0 or patch_w <= 0:
            raise ValueError("Patch dimensions must be positive.")

        if stride_h <= 0 or stride_w <= 0:
            raise ValueError("Stride dimensions must be positive.")

        if stride_h > patch_h or stride_w > patch_w:
            raise ValueError(
                "Stride must not exceed patch size."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "Batch size must be positive."
            )

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_everything(seed: int) -> None:
    """Set random seeds for reproducible inference."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
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
# IMAGE DISCOVERY
# =============================================================================

RUN_PATTERN = re.compile(
    r"run[\s_-]*(\d+)",
    re.IGNORECASE,
)


def is_mask_file(path: Path) -> bool:
    """Exclude files that appear to be masks or model outputs."""

    stem = path.stem.lower()

    tokens = (
        "_mask",
        "-mask",
        "_labels",
        "-labels",
        "_seg",
        "-seg",
        "_pred_mask",
        "_probability",
    )

    return any(
        stem.endswith(token)
        for token in tokens
    )


def discover_images(
    input_dir: Path,
    supported_extensions: Iterable[str],
) -> List[Path]:
    """Recursively find SEM images."""

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory not found:\n{input_dir}"
        )

    extensions = {
        ext.lower()
        for ext in supported_extensions
    }

    images = [
        path
        for path in input_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in extensions
            and not is_mask_file(path)
        )
    ]

    images.sort(
        key=lambda path: str(
            path.relative_to(input_dir)
        ).lower()
    )

    if not images:
        raise RuntimeError(
            f"No supported images found in:\n{input_dir}"
        )

    return images


def infer_run_name(
    path: Path,
    input_dir: Path,
) -> str:
    """Extract experimental Run ID from file or folder names."""

    candidates = [path.stem]

    candidates.extend(
        parent.name
        for parent in path.parents
        if parent != input_dir.parent
    )

    for text in candidates:
        match = RUN_PATTERN.search(text)

        if match:
            return f"Run{int(match.group(1))}"

    return "Unassigned"


def validate_unique_outputs(
    image_paths: Sequence[Path],
    input_dir: Path,
) -> None:
    """Ensure two input images cannot overwrite one output file."""

    seen: Dict[Tuple[str, str], Path] = {}

    for path in image_paths:
        run_name = infer_run_name(
            path,
            input_dir,
        )

        key = (
            run_name.lower(),
            path.stem.lower(),
        )

        if key in seen:
            raise RuntimeError(
                "Duplicate output identity detected:\n"
                f"{seen[key]}\n"
                f"{path}"
            )

        seen[key] = path


# =============================================================================
# SLIDING-WINDOW GEOMETRY
# =============================================================================

def sliding_positions(
    full_length: int,
    patch_length: int,
    stride: int,
) -> List[int]:
    """Generate patch positions with complete boundary coverage."""

    if full_length <= patch_length:
        return [0]

    positions = list(
        range(
            0,
            full_length - patch_length + 1,
            stride,
        )
    )

    last_position = (
        full_length - patch_length
    )

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions


def pad_image_if_needed(
    image: np.ndarray,
    patch_size: Tuple[int, int],
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad images smaller than a single inference patch."""

    original_h, original_w = image.shape
    patch_h, patch_w = patch_size

    pad_bottom = max(
        0,
        patch_h - original_h,
    )

    pad_right = max(
        0,
        patch_w - original_w,
    )

    if pad_bottom == 0 and pad_right == 0:
        return image, (
            original_h,
            original_w,
        )

    mode = (
        "reflect"
        if original_h > 1 and original_w > 1
        else "edge"
    )

    padded = np.pad(
        image,
        (
            (0, pad_bottom),
            (0, pad_right),
        ),
        mode=mode,
    )

    return padded, (
        original_h,
        original_w,
    )


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(
    checkpoint_path: Path,
    device: torch.device,
    config: InferenceConfig,
):
    """Load the best trained U-Net checkpoint."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
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

    checkpoint_threshold = float(
        training_config.get(
            "metric_threshold",
            config.threshold,
        )
    )

    if not np.isclose(
        checkpoint_threshold,
        config.threshold,
    ):
        raise RuntimeError(
            "Inference threshold differs from "
            "the checkpoint threshold."
        )

    model = build_unet(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        dropout_probability=dropout_probability,
        groups=groups,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    metadata = {
        "checkpoint_epoch": int(
            checkpoint.get(
                "epoch",
                -1,
            )
        ),
        "best_validation_dice": float(
            checkpoint.get(
                "best_validation_dice",
                float("nan"),
            )
        ),
        "base_channels": base_channels,
        "dropout_probability": dropout_probability,
        "groups": groups,
        "threshold": checkpoint_threshold,
    }

    return model, metadata


# =============================================================================
# FULL-IMAGE INFERENCE
# =============================================================================

@torch.inference_mode()
def predict_full_image(
    model: torch.nn.Module,
    normalized_image: np.ndarray,
    device: torch.device,
    config: InferenceConfig,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Predict one complete SEM image.

    Overlapping patch probabilities are averaged before thresholding.
    """

    if normalized_image.ndim != 2:
        raise ValueError(
            "Expected a 2D grayscale image."
        )

    padded_image, original_shape = (
        pad_image_if_needed(
            normalized_image,
            config.patch_size,
        )
    )

    original_h, original_w = original_shape
    padded_h, padded_w = padded_image.shape

    patch_h, patch_w = config.patch_size
    stride_h, stride_w = config.stride

    tops = sliding_positions(
        padded_h,
        patch_h,
        stride_h,
    )

    lefts = sliding_positions(
        padded_w,
        patch_w,
        stride_w,
    )

    locations = [
        (top, left)
        for top in tops
        for left in lefts
    ]

    probability_sum = np.zeros(
        (padded_h, padded_w),
        dtype=np.float64,
    )

    overlap_count = np.zeros(
        (padded_h, padded_w),
        dtype=np.uint16,
    )

    for start in range(
        0,
        len(locations),
        config.batch_size,
    ):
        batch_locations = locations[
            start:start + config.batch_size
        ]

        patches = np.stack(
            [
                padded_image[
                    top:top + patch_h,
                    left:left + patch_w,
                ]
                for top, left
                in batch_locations
            ]
        ).astype(
            np.float32,
            copy=False,
        )

        batch = torch.from_numpy(
            patches[:, None, :, :]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        logits = model(batch)

        probabilities = torch.sigmoid(
            logits
        )[:, 0]

        probabilities = (
            probabilities
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        for index, (top, left) in enumerate(
            batch_locations
        ):
            probability_sum[
                top:top + patch_h,
                left:left + patch_w,
            ] += probabilities[index]

            overlap_count[
                top:top + patch_h,
                left:left + patch_w,
            ] += 1

    if np.any(overlap_count == 0):
        raise RuntimeError(
            "Sliding-window reconstruction "
            "left uncovered pixels."
        )

    probability_map = (
        probability_sum
        / overlap_count.astype(np.float64)
    ).astype(np.float32)

    probability_map = probability_map[
        :original_h,
        :original_w,
    ]

    if not np.isfinite(
        probability_map
    ).all():
        raise RuntimeError(
            "Probability map contains NaN or Inf."
        )

    binary_mask = (
        probability_map
        >= config.threshold
    ).astype(np.uint8)

    return (
        probability_map,
        binary_mask,
        len(locations),
    )


# =============================================================================
# OUTPUTS
# =============================================================================

def save_prediction(
    output_dir: Path,
    run_name: str,
    sample_id: str,
    probability_map: np.ndarray,
    binary_mask: np.ndarray,
    config: InferenceConfig,
) -> Tuple[str, str]:
    """Save full-resolution mask and probability map."""

    mask_path_text = ""
    probability_path_text = ""

    if config.save_binary_masks:
        mask_dir = (
            output_dir
            / "predicted_masks"
            / run_name
        )

        mask_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        mask_path = (
            mask_dir
            / f"{sample_id}_pred_mask.tif"
        )

        tifffile.imwrite(
            mask_path,
            (
                binary_mask * 255
            ).astype(np.uint8),
            photometric="minisblack",
        )

        mask_path_text = str(
            mask_path
        )

    if config.save_probability_maps:
        probability_dir = (
            output_dir
            / "probability_maps"
            / run_name
        )

        probability_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        probability_path = (
            probability_dir
            / f"{sample_id}_probability.tif"
        )

        tifffile.imwrite(
            probability_path,
            probability_map.astype(
                np.float32
            ),
            photometric="minisblack",
        )

        probability_path_text = str(
            probability_path
        )

    return (
        mask_path_text,
        probability_path_text,
    )


def write_manifest(
    rows: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write image-level inference results."""

    path = (
        output_dir
        / "inference_manifest.csv"
    )

    fieldnames = [
        "image_index",
        "run",
        "sample_id",
        "source_image",
        "height",
        "width",
        "patch_count",
        "threshold",
        "predicted_zno_pixels",
        "total_pixels",
        "predicted_coverage_fraction",
        "predicted_coverage_percent",
        "mean_zno_probability",
        "processing_seconds",
        "predicted_mask_path",
        "probability_map_path",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def summarize_by_run(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Calculate coverage statistics for each experimental Run."""

    groups: Dict[str, List[float]] = {}

    for row in rows:
        groups.setdefault(
            str(row["run"]),
            [],
        ).append(
            float(
                row[
                    "predicted_coverage_percent"
                ]
            )
        )

    summary = {}

    for run_name, values in groups.items():
        array = np.asarray(
            values,
            dtype=float,
        )

        summary[run_name] = {
            "n_images": int(
                array.size
            ),
            "mean": float(
                array.mean()
            ),
            "std": float(
                array.std(ddof=1)
                if array.size > 1
                else 0.0
            ),
            "min": float(
                array.min()
            ),
            "max": float(
                array.max()
            ),
        }

    return summary


def write_metadata(
    output_dir: Path,
    config: InferenceConfig,
    checkpoint_path: Path,
    input_dir: Path,
    device: torch.device,
    checkpoint_metadata: Dict[str, Any],
) -> None:
    """Save inference configuration and model metadata."""

    metadata = {
        **asdict(config),
        "input_directory": str(
            input_dir
        ),
        "checkpoint": str(
            checkpoint_path
        ),
        "device": str(
            device
        ),
        **checkpoint_metadata,
    }

    path = (
        output_dir
        / "inference_config.json"
    )

    path.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )


# =============================================================================
# PIPELINE
# =============================================================================

def run_inference(
    input_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    limit: int | None = None,
) -> None:
    """Run full-dataset U-Net inference."""

    config = InferenceConfig()
    config.validate()

    seed_everything(
        config.seed
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = discover_images(
        input_dir,
        SUPPORTED_EXTENSIONS,
    )

    validate_unique_outputs(
        image_paths,
        input_dir,
    )

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "--limit must be positive."
            )

        image_paths = image_paths[
            :limit
        ]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, checkpoint_metadata = (
        load_model(
            checkpoint_path,
            device,
            config,
        )
    )

    print("=" * 72)
    print("FULL-DATASET U-NET INFERENCE")
    print("=" * 72)
    print(f"Images     : {len(image_paths)}")
    print(f"Device     : {device}")
    print(f"Patch size : {config.patch_size}")
    print(f"Stride     : {config.stride}")
    print(f"Threshold  : {config.threshold}")
    print("=" * 72)

    rows: List[Dict[str, Any]] = []

    total_start = time.perf_counter()

    for image_index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        start = time.perf_counter()

        run_name = infer_run_name(
            image_path,
            input_dir,
        )

        sample_id = image_path.stem

        raw_image = read_image(
            image_path
        )

        grayscale = to_grayscale(
            raw_image,
            image_path,
        )

        normalized = normalize_sem_image(
            grayscale
        )

        (
            probability_map,
            binary_mask,
            patch_count,
        ) = predict_full_image(
            model=model,
            normalized_image=normalized,
            device=device,
            config=config,
        )

        (
            mask_path,
            probability_path,
        ) = save_prediction(
            output_dir=output_dir,
            run_name=run_name,
            sample_id=sample_id,
            probability_map=probability_map,
            binary_mask=binary_mask,
            config=config,
        )

        predicted_pixels = int(
            binary_mask.sum()
        )

        total_pixels = int(
            binary_mask.size
        )

        coverage_fraction = (
            predicted_pixels
            / total_pixels
        )

        coverage_percent = (
            100.0
            * coverage_fraction
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        rows.append(
            {
                "image_index": image_index,
                "run": run_name,
                "sample_id": sample_id,
                "source_image": str(
                    image_path
                ),
                "height": int(
                    grayscale.shape[0]
                ),
                "width": int(
                    grayscale.shape[1]
                ),
                "patch_count": patch_count,
                "threshold": config.threshold,
                "predicted_zno_pixels": predicted_pixels,
                "total_pixels": total_pixels,
                "predicted_coverage_fraction": coverage_fraction,
                "predicted_coverage_percent": coverage_percent,
                "mean_zno_probability": float(
                    probability_map.mean()
                ),
                "processing_seconds": elapsed,
                "predicted_mask_path": mask_path,
                "probability_map_path": probability_path,
            }
        )

        # Preserve progress during long CPU inference runs.
        write_manifest(
            rows,
            output_dir,
        )

        print(
            f"[{image_index:03d}/{len(image_paths):03d}] "
            f"{run_name:<6} | "
            f"{sample_id:<30} | "
            f"coverage={coverage_percent:7.3f}% | "
            f"{elapsed:6.2f}s"
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    write_manifest(
        rows,
        output_dir,
    )

    write_metadata(
        output_dir=output_dir,
        config=config,
        checkpoint_path=checkpoint_path,
        input_dir=input_dir,
        device=device,
        checkpoint_metadata=checkpoint_metadata,
    )

    run_summary = summarize_by_run(
        rows
    )

    print("\n" + "=" * 72)
    print("INFERENCE COMPLETE")
    print("=" * 72)

    for run_name in sorted(
        run_summary
    ):
        values = run_summary[
            run_name
        ]

        print(
            f"{run_name:<8} "
            f"n={values['n_images']:<3} | "
            f"coverage="
            f"{values['mean']:7.3f} ± "
            f"{values['std']:7.3f}%"
        )

    print(
        f"\nTotal time: {total_seconds:.2f} s"
    )

    print(
        f"Outputs: {output_dir}"
    )


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the trained U-Net to the "
            "complete 10 µm SEM dataset."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing SEM images.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to best_model.pth.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for inference outputs.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of images to process "
            "for a smoke test."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    run_inference(
        input_dir=args.input_dir.expanduser().resolve(),
        checkpoint_path=args.checkpoint.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
