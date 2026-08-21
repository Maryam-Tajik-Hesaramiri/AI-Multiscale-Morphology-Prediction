#!/usr/bin/env python
"""
audit_training_distribution.py
===================================

Audit foreground coverage and training-patch sampling for the 10 µm U-Net segmentation dataset.

Checks:
1. Foreground coverage for every image in Train / Val / Test.
2. Per-run coverage statistics.
3. Actual training-patch foreground distribution using the selected
   TrainingPatchDataset sampling logic.
4. Fractions of mostly-background, mixed, mostly-foreground, and
   nearly-all-foreground patches.

No dataset files or existing model outputs are modified.\n\nPortfolio version uses the cleaned prepare_dataloader.py module and runtime-configurable paths.\n"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATALOADER_FILE = PROJECT_DIR / "prepare_dataloader.py"

OUTPUT_DIR = PROJECT_DIR / "training_distribution_audit"

IMAGE_CSV = OUTPUT_DIR / "image_foreground_coverage.csv"
RUN_CSV = OUTPUT_DIR / "run_foreground_summary.csv"
PATCH_CSV = OUTPUT_DIR / "training_patch_distribution.csv"
SUMMARY_TXT = OUTPUT_DIR / "audit_summary.txt"


def parse_arguments() -> argparse.Namespace:
    """Parse optional paths for the standalone audit."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit image-level foreground coverage and the actual "
            "training-patch sampling distribution."
        )
    )
    parser.add_argument(
        "--dataloader-file",
        type=Path,
        default=DATALOADER_FILE,
        help="Path to prepare_dataloader.py.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Optional dataset root. If provided, it overrides the "
            "DataLoader module's default dataset location."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for audit CSV and text outputs.",
    )
    return parser.parse_args()


# =============================================================================
# 2. LOAD EXISTING DATALOADER MODULE
# =============================================================================

def load_module(module_name: str, file_path: Path):
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


# =============================================================================
# 3. RUN IDENTIFICATION
# =============================================================================

def extract_run(sample_id: str) -> str:
    """
    Extract Run number from names such as:
        Run5_5um-Image1_008
    """
    match = re.search(r"run\s*[_-]?(\d+)", sample_id, flags=re.I)

    if match:
        return f"Run{int(match.group(1))}"

    return "Unknown"


# =============================================================================
# 4. IMAGE-LEVEL COVERAGE
# =============================================================================

def audit_image_coverage(dl):
    rows = []

    for split_name in ("train", "val", "test"):
        pairs = dl.discover_pairs(split_name)

        for pair in pairs:
            mask = dl.prepare_binary_mask(
                dl.read_image(pair.mask_path),
                pair.mask_path,
            )

            foreground_fraction = float(mask.mean())

            rows.append(
                {
                    "split": split_name,
                    "run": extract_run(pair.sample_id),
                    "sample_id": pair.sample_id,
                    "height": int(mask.shape[0]),
                    "width": int(mask.shape[1]),
                    "foreground_fraction": foreground_fraction,
                    "foreground_percent": 100.0 * foreground_fraction,
                }
            )

    return rows


# =============================================================================
# 5. RUN-LEVEL SUMMARY
# =============================================================================

def summarize_runs(image_rows):
    grouped = defaultdict(list)

    for row in image_rows:
        key = (row["split"], row["run"])
        grouped[key].append(row["foreground_fraction"])

    rows = []

    for (split_name, run_name), values in sorted(grouped.items()):
        values = np.asarray(values, dtype=float)

        rows.append(
            {
                "split": split_name,
                "run": run_name,
                "n_images": len(values),
                "mean_foreground_fraction": float(values.mean()),
                "std_foreground_fraction": float(
                    values.std(ddof=1) if len(values) > 1 else 0.0
                ),
                "min_foreground_fraction": float(values.min()),
                "max_foreground_fraction": float(values.max()),
                "mean_foreground_percent": float(values.mean() * 100.0),
            }
        )

    return rows


# =============================================================================
# 6. TRAINING PATCH AUDIT
# =============================================================================

def classify_patch(fraction: float) -> str:
    """
    Coarse categories chosen only for diagnostic interpretation.
    """
    if fraction < 0.01:
        return "background_<1%"
    if fraction < 0.25:
        return "low_fg_1-25%"
    if fraction < 0.75:
        return "mixed_25-75%"
    if fraction < 0.99:
        return "high_fg_75-99%"
    return "almost_all_fg_>=99%"


def audit_training_patches(dl):
    train_pairs = dl.discover_pairs("train")

    dataset = dl.TrainingPatchDataset(
        pairs=train_pairs,
        patch_size=dl.PATCH_SIZE,
        patches_per_image=dl.TRAIN_PATCHES_PER_IMAGE,
        seed=dl.SEED,
    )

    rows = []

    # Evaluate several epochs because training coordinates change every epoch.
    audit_epochs = [1, 2, 3, 4, 5]

    for epoch in audit_epochs:
        dataset.set_epoch(epoch)

        for index in range(len(dataset)):
            sample = dataset[index]

            mask = sample["mask"].numpy().squeeze()

            foreground_fraction = float(mask.mean())

            rows.append(
                {
                    "epoch": epoch,
                    "run": extract_run(sample["sample_id"]),
                    "sample_id": sample["sample_id"],
                    "top": int(sample["top"]),
                    "left": int(sample["left"]),
                    "foreground_fraction": foreground_fraction,
                    "foreground_percent": foreground_fraction * 100.0,
                    "category": classify_patch(foreground_fraction),
                }
            )

    return rows


# =============================================================================
# 7. CSV WRITING
# =============================================================================

def write_csv(path: Path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# 8. SUMMARY
# =============================================================================

def build_summary(image_rows, run_rows, patch_rows):
    categories = defaultdict(int)

    for row in patch_rows:
        categories[row["category"]] += 1

    total_patches = len(patch_rows)

    train_image_values = np.asarray(
        [
            row["foreground_fraction"]
            for row in image_rows
            if row["split"] == "train"
        ],
        dtype=float,
    )

    val_image_values = np.asarray(
        [
            row["foreground_fraction"]
            for row in image_rows
            if row["split"] == "val"
        ],
        dtype=float,
    )

    test_image_values = np.asarray(
        [
            row["foreground_fraction"]
            for row in image_rows
            if row["split"] == "test"
        ],
        dtype=float,
    )

    patch_values = np.asarray(
        [row["foreground_fraction"] for row in patch_rows],
        dtype=float,
    )

    lines = [
        "=" * 78,
        "10 µm U-NET TRAINING DISTRIBUTION AUDIT",
        "=" * 78,
        "",
        "IMAGE-LEVEL FOREGROUND COVERAGE",
        "-" * 78,
        f"Train images: {len(train_image_values)}",
        f"  Mean foreground: {train_image_values.mean() * 100:.2f}%",
        f"  Min foreground:  {train_image_values.min() * 100:.2f}%",
        f"  Max foreground:  {train_image_values.max() * 100:.2f}%",
        "",
        f"Validation images: {len(val_image_values)}",
        f"  Mean foreground: {val_image_values.mean() * 100:.2f}%",
        f"  Min foreground:  {val_image_values.min() * 100:.2f}%",
        f"  Max foreground:  {val_image_values.max() * 100:.2f}%",
        "",
        f"Test images: {len(test_image_values)}",
        f"  Mean foreground: {test_image_values.mean() * 100:.2f}%",
        f"  Min foreground:  {test_image_values.min() * 100:.2f}%",
        f"  Max foreground:  {test_image_values.max() * 100:.2f}%",
        "",
        "ACTUAL TRAINING PATCH DISTRIBUTION",
        "-" * 78,
        f"Audited patches: {total_patches}",
        f"Mean patch foreground: {patch_values.mean() * 100:.2f}%",
        f"Median patch foreground: {np.median(patch_values) * 100:.2f}%",
        "",
    ]

    category_order = [
        "background_<1%",
        "low_fg_1-25%",
        "mixed_25-75%",
        "high_fg_75-99%",
        "almost_all_fg_>=99%",
    ]

    for category in category_order:
        count = categories.get(category, 0)
        percent = (
            100.0 * count / total_patches
            if total_patches > 0
            else 0.0
        )

        lines.append(
            f"{category:<24} "
            f"{count:>5} patches "
            f"({percent:6.2f}%)"
        )

    lines.extend(
        [
            "",
            "RUN-LEVEL OUTPUT",
            "-" * 78,
            f"Run summary rows: {len(run_rows)}",
            "",
            "Interpretation note:",
            "This audit does not judge segmentation quality. It checks whether",
            "the current training sampling distribution is strongly biased toward",
            "foreground-heavy patches before any retraining decisions are made.",
            "=" * 78,
        ]
    )

    return "\n".join(lines)


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    global OUTPUT_DIR
    global IMAGE_CSV
    global RUN_CSV
    global PATCH_CSV
    global SUMMARY_TXT

    args = parse_arguments()

    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    IMAGE_CSV = OUTPUT_DIR / "image_foreground_coverage.csv"
    RUN_CSV = OUTPUT_DIR / "run_foreground_summary.csv"
    PATCH_CSV = OUTPUT_DIR / "training_patch_distribution.csv"
    SUMMARY_TXT = OUTPUT_DIR / "audit_summary.txt"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dl = load_module(
        "prepare_dataloader_audit",
        args.dataloader_file.expanduser().resolve(),
    )

    if args.dataset_dir is not None:
        if hasattr(dl, "set_dataset_dir"):
            dl.set_dataset_dir(args.dataset_dir)
        else:
            dl.DATASET_DIR = args.dataset_dir.expanduser().resolve()

    print("=" * 78)
    print("TRAINING DISTRIBUTION AUDIT")
    print("=" * 78)

    print("Analyzing full-image foreground coverage...")
    image_rows = audit_image_coverage(dl)

    print("Summarizing coverage by Run...")
    run_rows = summarize_runs(image_rows)

    print("Auditing actual training-patch sampling...")
    patch_rows = audit_training_patches(dl)

    write_csv(IMAGE_CSV, image_rows)
    write_csv(RUN_CSV, run_rows)
    write_csv(PATCH_CSV, patch_rows)

    summary = build_summary(
        image_rows=image_rows,
        run_rows=run_rows,
        patch_rows=patch_rows,
    )

    SUMMARY_TXT.write_text(
        summary,
        encoding="utf-8",
    )

    print("")
    print(summary)
    print("")
    print("Audit files saved successfully.")


if __name__ == "__main__":
    main()
