# ============================================================
# AI-driven Multiscale Morphology Prediction
#
# Phase 2.1 — 10 µm Binary Segmentation
#
# Script:
# dataset_split.py
#
# Purpose:
# Create a reproducible Run-balanced and coverage-balanced
# Train / Validation / Test split.
#
# Split per Run:
#   Train      = 3 image-mask pairs
#   Validation = 1 image-mask pair
#   Test       = 1 image-mask pair
#
# The split optimizer:
# - Preserves all image-mask pairs
# - Keeps every Run in every split
# - Uses mask coverage to improve distribution balance
# - Searches many possible 3/1/1 assignments
# - Selects the assignment with the best coverage balance
#
# Classes:
#   0 = Al
#   1 = ZnO
#
# Important:
# - Original files are never moved or modified.
# - Only manually labeled images are included.
# - Source images without masks are ignored.
# - The fixed random seed makes the split reproducible.
#
# Author: Maryam Tajik Hesaramiri
# Portfolio version: local paths removed; dataset paths are supplied at runtime.
# ============================================================
#
# Example:
#   python dataset_split.py \
#       --image-root /path/to/images \
#       --mask-root /path/to/manual_masks
#
# ============================================================

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# These are configured from command-line arguments in main().
IMAGE_ROOT = Path(".")
MASK_ROOT = Path(".")

OUTPUT_ROOT = PROJECT_ROOT / "dataset_split_10um"
SUMMARY_CSV = PROJECT_ROOT / "split_summary.csv"
STATISTICS_TXT = PROJECT_ROOT / "split_statistics.txt"

RANDOM_SEED = 42

RUN_NAMES = [f"Run{i}" for i in range(1, 10)]

EXPECTED_PAIRS_PER_RUN = 5

TRAIN_PER_RUN = 3
VAL_PER_RUN = 1
TEST_PER_RUN = 1

TIFF_EXTENSIONS = {".tif", ".tiff"}

# Number of candidate splits tested by the optimizer.
OPTIMIZATION_ITERATIONS = 200_000

# Safety default. Can be enabled with --overwrite.
OVERWRITE_EXISTING_OUTPUT = False


def parse_arguments() -> argparse.Namespace:
    """Parse dataset locations without embedding machine-specific paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible run-balanced and coverage-balanced "
            "Train/Validation/Test split for labeled SEM images."
        )
    )

    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help=(
            "Root directory containing Run1 ... Run9 image folders."
        ),
    )

    parser.add_argument(
        "--mask-root",
        type=Path,
        required=True,
        help=(
            "Root directory containing Run1 ... Run9 folders, "
            "each with a masks/ subfolder."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional output directory for the split dataset. "
            "Default: <script_dir>/dataset_split_10um"
        ),
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=OPTIMIZATION_ITERATIONS,
        help="Number of candidate split assignments to evaluate.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild an existing output directory.",
    )

    return parser.parse_args()


def configure_runtime_paths(args: argparse.Namespace) -> None:
    """Apply command-line configuration to the existing pipeline."""
    global IMAGE_ROOT
    global MASK_ROOT
    global OUTPUT_ROOT
    global SUMMARY_CSV
    global STATISTICS_TXT
    global OPTIMIZATION_ITERATIONS
    global OVERWRITE_EXISTING_OUTPUT

    IMAGE_ROOT = args.image_root.expanduser().resolve()
    MASK_ROOT = args.mask_root.expanduser().resolve()

    if args.output_root is None:
        OUTPUT_ROOT = PROJECT_ROOT / "dataset_split_10um"
    else:
        OUTPUT_ROOT = args.output_root.expanduser().resolve()

    report_root = OUTPUT_ROOT.parent
    SUMMARY_CSV = report_root / "split_summary.csv"
    STATISTICS_TXT = report_root / "split_statistics.txt"

    if args.iterations <= 0:
        raise ValueError("--iterations must be a positive integer.")

    OPTIMIZATION_ITERATIONS = int(args.iterations)
    OVERWRITE_EXISTING_OUTPUT = bool(args.overwrite)


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass(frozen=True)
class Sample:
    """
    One paired image-mask sample.
    """

    run_name: str
    pairing_key: str
    image_path: Path
    mask_path: Path
    coverage_percent: float


# ============================================================
# FILENAME NORMALIZATION
# ============================================================

def normalize_stem(path: Path) -> str:
    """
    Normalize image and Napari mask filenames to a common key.

    Examples
    --------
    Image:
        5um-Image1_001.tif
        -> 5um-image1_001

    Mask:
        5um-Image1_001 - Labels.tif
        -> 5um-image1_001
    """
    stem = path.stem.strip().lower()

    suffixes = sorted(
        (
            " - labels",
            "- labels",
            "_labels",
            "-labels",
            " labels",
            " - label",
            "- label",
            "_label",
            "-label",
            " label",
            " - masks",
            "- masks",
            "_masks",
            "-masks",
            " masks",
            " - mask",
            "- mask",
            "_mask",
            "-mask",
            " mask",
        ),
        key=len,
        reverse=True,
    )

    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)].strip()
            break

    return stem


# ============================================================
# FILE SEARCH AND PAIRING
# ============================================================

def find_tiff_files(folder: Path) -> list[Path]:
    """
    Find TIFF files directly inside a folder.
    """
    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found:\n{folder}"
        )

    if not folder.is_dir():
        raise NotADirectoryError(
            f"Expected a directory but found:\n{folder}"
        )

    files = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in TIFF_EXTENSIONS
    ]

    return sorted(
        files,
        key=lambda path: path.name.lower(),
    )


def build_unique_file_map(
    files: list[Path],
    file_type: str,
    run_name: str,
) -> dict[str, Path]:
    """
    Build a normalized filename-to-path mapping.
    """
    file_map: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for file_path in files:
        pairing_key = normalize_stem(file_path)

        if not pairing_key:
            raise RuntimeError(
                f"Could not generate a pairing key for:\n"
                f"{file_path}"
            )

        if pairing_key in file_map:
            duplicates.setdefault(
                pairing_key,
                [file_map[pairing_key]],
            ).append(file_path)
        else:
            file_map[pairing_key] = file_path

    if duplicates:
        lines = [
            f"Duplicate normalized {file_type} names "
            f"were found in {run_name}."
        ]

        for pairing_key, duplicate_paths in duplicates.items():
            lines.append(
                f"\nPairing key: {pairing_key}"
            )

            for duplicate_path in duplicate_paths:
                lines.append(
                    f"  - {duplicate_path.name}"
                )

        raise RuntimeError("\n".join(lines))

    return file_map


def pair_images_and_masks(
    image_folder: Path,
    mask_folder: Path,
    run_name: str,
) -> list[tuple[Path, Path]]:
    """
    Pair manually created masks with original SEM images.

    Masks define which images belong to the labeled dataset.
    Source images without masks are intentionally ignored.
    """
    image_files = find_tiff_files(image_folder)
    mask_files = find_tiff_files(mask_folder)

    if not image_files:
        raise RuntimeError(
            f"No TIFF images were found for {run_name}:\n"
            f"{image_folder}"
        )

    if not mask_files:
        raise RuntimeError(
            f"No TIFF masks were found for {run_name}:\n"
            f"{mask_folder}"
        )

    image_map = build_unique_file_map(
        files=image_files,
        file_type="image",
        run_name=run_name,
    )

    mask_map = build_unique_file_map(
        files=mask_files,
        file_type="mask",
        run_name=run_name,
    )

    pairs: list[tuple[Path, Path]] = []
    masks_without_images: list[Path] = []

    for pairing_key, mask_path in sorted(mask_map.items()):
        image_path = image_map.get(pairing_key)

        if image_path is None:
            masks_without_images.append(mask_path)
        else:
            pairs.append((image_path, mask_path))

    if masks_without_images:
        lines = [
            f"Some masks in {run_name} do not have matching images:"
        ]

        for mask_path in masks_without_images:
            lines.append(f"  - {mask_path.name}")

        lines.extend(
            [
                "",
                "Expected naming example:",
                "  Image: 5um-Image1_001.tif",
                "  Mask:  5um-Image1_001 - Labels.tif",
            ]
        )

        raise RuntimeError("\n".join(lines))

    return pairs


# ============================================================
# MASK VALIDATION AND COVERAGE
# ============================================================

def load_tiff(path: Path) -> np.ndarray:
    """
    Read a TIFF file into a NumPy array.
    """
    try:
        with Image.open(path) as image:
            array = np.asarray(image)
    except Exception as error:
        raise RuntimeError(
            f"Could not read TIFF file:\n{path}\n\n"
            f"Original error: {error}"
        ) from error

    if array.size == 0:
        raise RuntimeError(
            f"Empty TIFF array detected:\n{path}"
        )

    return array


def mask_to_binary(
    mask_array: np.ndarray,
    mask_path: Path,
) -> np.ndarray:
    """
    Validate a mask and convert it to binary 0/1.
    """
    if mask_array.ndim == 3:
        first_channel = mask_array[..., 0]

        identical_channels = all(
            np.array_equal(
                first_channel,
                mask_array[..., channel_index],
            )
            for channel_index in range(
                1,
                mask_array.shape[2],
            )
        )

        if not identical_channels:
            raise RuntimeError(
                f"Mask contains non-identical channels:\n"
                f"{mask_path}\n"
                f"Shape: {mask_array.shape}"
            )

        mask_array = first_channel

    if mask_array.ndim != 2:
        raise RuntimeError(
            f"Mask must be two-dimensional:\n"
            f"{mask_path}\n"
            f"Shape: {mask_array.shape}"
        )

    unique_values = {
        int(value)
        for value in np.unique(mask_array)
    }

    valid_sets = (
        {0},
        {1},
        {255},
        {0, 1},
        {0, 255},
    )

    if unique_values not in valid_sets:
        raise RuntimeError(
            f"Mask is not binary:\n"
            f"{mask_path}\n"
            f"Unique values: {sorted(unique_values)}"
        )

    return (mask_array > 0).astype(np.uint8)


def calculate_mask_coverage(mask_path: Path) -> float:
    """
    Calculate ZnO foreground coverage percentage.
    """
    mask_array = load_tiff(mask_path)
    binary_mask = mask_to_binary(
        mask_array=mask_array,
        mask_path=mask_path,
    )

    coverage = (
        np.count_nonzero(binary_mask)
        / binary_mask.size
    ) * 100.0

    return float(coverage)


def create_samples(
    all_run_pairs: dict[str, list[tuple[Path, Path]]],
) -> dict[str, list[Sample]]:
    """
    Calculate coverage for every labeled sample.
    """
    samples_by_run: dict[str, list[Sample]] = {}

    print("\nCalculating mask coverage before split optimization...")

    for run_name in RUN_NAMES:
        run_samples: list[Sample] = []

        for image_path, mask_path in all_run_pairs[run_name]:
            coverage = calculate_mask_coverage(mask_path)

            run_samples.append(
                Sample(
                    run_name=run_name,
                    pairing_key=normalize_stem(image_path),
                    image_path=image_path,
                    mask_path=mask_path,
                    coverage_percent=coverage,
                )
            )

            print(
                f"  {run_name}: {image_path.name} — "
                f"Coverage={coverage:.2f}%"
            )

        samples_by_run[run_name] = run_samples

    return samples_by_run


# ============================================================
# COVERAGE-BALANCED SPLIT OPTIMIZATION
# ============================================================

def descriptive_vector(values: list[float]) -> np.ndarray:
    """
    Return normalized descriptive features for an objective function.

    Features:
    - Mean
    - Standard deviation
    - Minimum
    - 10th percentile
    - 25th percentile
    - Median
    - 75th percentile
    - 90th percentile
    - Maximum
    """
    array = np.asarray(values, dtype=np.float64)

    return np.asarray(
        [
            np.mean(array),
            np.std(array, ddof=0),
            np.min(array),
            np.percentile(array, 10),
            np.percentile(array, 25),
            np.median(array),
            np.percentile(array, 75),
            np.percentile(array, 90),
            np.max(array),
        ],
        dtype=np.float64,
    )


def coverage_bin_proportions(
    values: list[float],
) -> np.ndarray:
    """
    Calculate proportions in broad coverage ranges.

    Bins:
      0–25%
      25–50%
      50–75%
      75–90%
      90–100%
    """
    array = np.asarray(values, dtype=np.float64)

    bins = np.asarray(
        [0.0, 25.0, 50.0, 75.0, 90.0, 100.000001],
        dtype=np.float64,
    )

    counts, _ = np.histogram(array, bins=bins)

    return counts.astype(np.float64) / array.size


def calculate_balance_score(
    train_values: list[float],
    val_values: list[float],
    test_values: list[float],
    overall_values: list[float],
) -> float:
    """
    Score one candidate split.

    Lower score means better agreement between Train, Validation,
    Test, and the overall dataset coverage distribution.
    """
    overall_description = descriptive_vector(overall_values)
    overall_bins = coverage_bin_proportions(overall_values)

    scale = np.asarray(
        [
            10.0,  # mean
            10.0,  # standard deviation
            15.0,  # minimum
            15.0,  # 10th percentile
            15.0,  # 25th percentile
            10.0,  # median
            10.0,  # 75th percentile
            10.0,  # 90th percentile
            10.0,  # maximum
        ],
        dtype=np.float64,
    )

    score = 0.0

    split_values = {
        "train": train_values,
        "val": val_values,
        "test": test_values,
    }

    # Validation and Test receive slightly greater weight because
    # they contain fewer samples and are used for model evaluation.
    split_weights = {
        "train": 1.0,
        "val": 1.5,
        "test": 1.5,
    }

    for split_name, values in split_values.items():
        description = descriptive_vector(values)
        normalized_difference = (
            description - overall_description
        ) / scale

        description_score = float(
            np.mean(normalized_difference ** 2)
        )

        bin_difference = (
            coverage_bin_proportions(values)
            - overall_bins
        )

        bin_score = float(
            np.mean(bin_difference ** 2)
        )

        score += split_weights[split_name] * (
            description_score
            + 3.0 * bin_score
        )

    # Directly encourage similar means among the three splits.
    split_means = np.asarray(
        [
            np.mean(train_values),
            np.mean(val_values),
            np.mean(test_values),
        ],
        dtype=np.float64,
    )

    mean_spread_penalty = float(
        np.std(split_means) ** 2
        / 100.0
    )

    score += 2.0 * mean_spread_penalty

    # Encourage low-coverage representation in Validation and Test
    # when low-coverage samples exist in the dataset.
    low_threshold = 25.0

    overall_low_count = sum(
        value < low_threshold
        for value in overall_values
    )

    if overall_low_count >= 2:
        val_has_low = any(
            value < low_threshold
            for value in val_values
        )

        test_has_low = any(
            value < low_threshold
            for value in test_values
        )

        if not val_has_low:
            score += 5.0

        if not test_has_low:
            score += 5.0

    return score


def generate_run_assignments(
    run_samples: list[Sample],
) -> list[dict[str, list[Sample]]]:
    """
    Generate all valid 3/1/1 assignments for one five-sample Run.

    There are:
        5 choices for Validation
        4 remaining choices for Test
        3 remaining samples for Train

    Total = 20 possible assignments per Run.
    """
    if len(run_samples) != EXPECTED_PAIRS_PER_RUN:
        raise ValueError(
            f"Expected 5 samples but received {len(run_samples)}."
        )

    assignments: list[dict[str, list[Sample]]] = []

    sample_indices = range(len(run_samples))

    for val_index, test_index in permutations(
        sample_indices,
        2,
    ):
        train_samples = [
            run_samples[index]
            for index in sample_indices
            if index not in {val_index, test_index}
        ]

        assignments.append(
            {
                "train": train_samples,
                "val": [run_samples[val_index]],
                "test": [run_samples[test_index]],
            }
        )

    return assignments


def optimize_split(
    samples_by_run: dict[str, list[Sample]],
) -> tuple[
    dict[str, list[Sample]],
    float,
]:
    """
    Search for a Run-balanced split with improved coverage balance.
    """
    assignments_by_run = {
        run_name: generate_run_assignments(
            samples_by_run[run_name]
        )
        for run_name in RUN_NAMES
    }

    all_samples = [
        sample
        for run_name in RUN_NAMES
        for sample in samples_by_run[run_name]
    ]

    overall_values = [
        sample.coverage_percent
        for sample in all_samples
    ]

    random_generator = random.Random(RANDOM_SEED)

    best_score = float("inf")
    best_split: dict[str, list[Sample]] | None = None

    print(
        f"\nSearching {OPTIMIZATION_ITERATIONS:,} candidate "
        f"Run-balanced splits..."
    )

    for iteration in range(1, OPTIMIZATION_ITERATIONS + 1):
        candidate_split = {
            "train": [],
            "val": [],
            "test": [],
        }

        for run_name in RUN_NAMES:
            run_assignment = random_generator.choice(
                assignments_by_run[run_name]
            )

            for split_name in (
                "train",
                "val",
                "test",
            ):
                candidate_split[split_name].extend(
                    run_assignment[split_name]
                )

        train_values = [
            sample.coverage_percent
            for sample in candidate_split["train"]
        ]

        val_values = [
            sample.coverage_percent
            for sample in candidate_split["val"]
        ]

        test_values = [
            sample.coverage_percent
            for sample in candidate_split["test"]
        ]

        score = calculate_balance_score(
            train_values=train_values,
            val_values=val_values,
            test_values=test_values,
            overall_values=overall_values,
        )

        if score < best_score:
            best_score = score
            best_split = {
                split_name: list(samples)
                for split_name, samples
                in candidate_split.items()
            }

        if iteration % 50_000 == 0:
            print(
                f"  Tested {iteration:,} candidates — "
                f"best score={best_score:.6f}"
            )

    if best_split is None:
        raise RuntimeError(
            "Split optimization failed to produce a result."
        )

    return best_split, best_score


# ============================================================
# OUTPUT CREATION
# ============================================================

def prepare_output_directories() -> dict[str, dict[str, Path]]:
    """
    Create Train / Validation / Test output directories.
    """
    PROJECT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_ROOT.exists():
        if OVERWRITE_EXISTING_OUTPUT:
            print(
                "\nDeleting existing output directory:"
            )
            print(OUTPUT_ROOT)

            shutil.rmtree(OUTPUT_ROOT)
        else:
            raise FileExistsError(
                "\nThe output directory already exists:\n"
                f"{OUTPUT_ROOT}\n\n"
                "Delete the existing output folder or rerun with --overwrite."
            )

    output_directories: dict[str, dict[str, Path]] = {}

    for split_name in (
        "train",
        "val",
        "test",
    ):
        image_folder = (
            OUTPUT_ROOT
            / split_name
            / "images"
        )

        mask_folder = (
            OUTPUT_ROOT
            / split_name
            / "masks"
        )

        image_folder.mkdir(
            parents=True,
            exist_ok=False,
        )

        mask_folder.mkdir(
            parents=True,
            exist_ok=False,
        )

        output_directories[split_name] = {
            "images": image_folder,
            "masks": mask_folder,
        }

    return output_directories


def make_output_filename(
    run_name: str,
    source_file: Path,
) -> str:
    """
    Add the Run name to avoid filename collisions.
    """
    return f"{run_name}_{source_file.name}"


def copy_sample(
    sample: Sample,
    split_name: str,
    output_directories: dict[str, dict[str, Path]],
) -> tuple[Path, Path]:
    """
    Copy one paired image and mask.
    """
    image_destination = (
        output_directories[split_name]["images"]
        / make_output_filename(
            sample.run_name,
            sample.image_path,
        )
    )

    mask_destination = (
        output_directories[split_name]["masks"]
        / make_output_filename(
            sample.run_name,
            sample.mask_path,
        )
    )

    shutil.copy2(
        sample.image_path,
        image_destination,
    )

    shutil.copy2(
        sample.mask_path,
        mask_destination,
    )

    return image_destination, mask_destination


# ============================================================
# REPORTS
# ============================================================

def summarize_coverage(
    values: list[float],
) -> dict[str, float]:
    """
    Calculate coverage summary statistics.
    """
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "count": float(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(
            np.std(array, ddof=1)
            if array.size > 1
            else 0.0
        ),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def write_summary_csv(
    records: list[dict[str, object]],
) -> None:
    """
    Write split membership and coverage for every pair.
    """
    fieldnames = [
        "run",
        "split",
        "pair_key",
        "coverage_percent",
        "source_image_name",
        "source_mask_name",
        "output_image_name",
        "output_mask_name",
    ]

    with SUMMARY_CSV.open(
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)


def write_statistics_txt(
    records: list[dict[str, object]],
    optimization_score: float,
) -> None:
    """
    Write split and coverage summary.
    """
    split_counts = Counter(
        str(record["split"])
        for record in records
    )

    run_split_counts: dict[
        str,
        Counter[str]
    ] = {
        run_name: Counter()
        for run_name in RUN_NAMES
    }

    coverage_by_split: dict[str, list[float]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for record in records:
        run_name = str(record["run"])
        split_name = str(record["split"])

        run_split_counts[run_name][split_name] += 1

        coverage_by_split[split_name].append(
            float(record["coverage_percent"])
        )

    overall_values = [
        float(record["coverage_percent"])
        for record in records
    ]

    summaries = {
        split_name: summarize_coverage(values)
        for split_name, values
        in coverage_by_split.items()
    }

    overall_summary = summarize_coverage(
        overall_values
    )

    lines = [
        "=" * 76,
        "AI-driven Multiscale Morphology Prediction",
        "Phase 2.1 — Coverage-Balanced 10 µm Dataset Split",
        "=" * 76,
        "",
        f"Random seed: {RANDOM_SEED}",
        f"Optimization iterations: {OPTIMIZATION_ITERATIONS:,}",
        f"Final optimization score: {optimization_score:.8f}",
        "",
        "Dataset paths are supplied at runtime and are not stored in this report.",
        "",
        f"Total Runs: {len(RUN_NAMES)}",
        f"Total image-mask pairs: {len(records)}",
        "",
        "Overall split counts:",
        f"  Train:      {split_counts['train']}",
        f"  Validation: {split_counts['val']}",
        f"  Test:       {split_counts['test']}",
        "",
        "Per-Run split:",
    ]

    for run_name in RUN_NAMES:
        counts = run_split_counts[run_name]

        lines.append(
            f"  {run_name}: "
            f"Train={counts['train']}, "
            f"Validation={counts['val']}, "
            f"Test={counts['test']}"
        )

    lines.append("")
    lines.append("Coverage statistics:")

    display_names = {
        "train": "Train",
        "val": "Validation",
        "test": "Test",
    }

    for split_name in (
        "train",
        "val",
        "test",
    ):
        summary = summaries[split_name]

        lines.extend(
            [
                "",
                f"  {display_names[split_name]}:",
                f"    Count:  {int(summary['count'])}",
                f"    Mean:   {summary['mean']:.4f} %",
                f"    Median: {summary['median']:.4f} %",
                f"    Std:    {summary['std']:.4f} %",
                f"    Min:    {summary['min']:.4f} %",
                f"    Max:    {summary['max']:.4f} %",
            ]
        )

    lines.extend(
        [
            "",
            "  Overall dataset:",
            f"    Count:  {int(overall_summary['count'])}",
            f"    Mean:   {overall_summary['mean']:.4f} %",
            f"    Median: {overall_summary['median']:.4f} %",
            f"    Std:    {overall_summary['std']:.4f} %",
            f"    Min:    {overall_summary['min']:.4f} %",
            f"    Max:    {overall_summary['max']:.4f} %",
            "",
            "Processing notes:",
            "- Original files were not modified or moved.",
            "- Only manually labeled image-mask pairs were used.",
            "- Every Run is represented in every split.",
            "- The 3/1/1 split was preserved for every Run.",
            "- Mask coverage was used to improve split balance.",
            "- The optimizer used a fixed random seed.",
            "",
            "=" * 76,
        ]
    )

    STATISTICS_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

def verify_final_split(
    selected_split: dict[str, list[Sample]],
) -> None:
    """
    Verify counts, uniqueness, and Run representation.
    """
    expected_counts = {
        "train": 27,
        "val": 9,
        "test": 9,
    }

    all_pairing_keys: list[str] = []

    for split_name, expected_count in expected_counts.items():
        samples = selected_split[split_name]

        if len(samples) != expected_count:
            raise RuntimeError(
                f"Incorrect count in {split_name}.\n"
                f"Expected: {expected_count}\n"
                f"Found:    {len(samples)}"
            )

        all_pairing_keys.extend(
            f"{sample.run_name}:{sample.pairing_key}"
            for sample in samples
        )

    if len(all_pairing_keys) != len(set(all_pairing_keys)):
        raise RuntimeError(
            "A sample appears in more than one split."
        )

    if len(all_pairing_keys) != 45:
        raise RuntimeError(
            f"Expected 45 unique samples but found "
            f"{len(all_pairing_keys)}."
        )

    for run_name in RUN_NAMES:
        run_counts = {
            split_name: sum(
                sample.run_name == run_name
                for sample in selected_split[split_name]
            )
            for split_name in (
                "train",
                "val",
                "test",
            )
        }

        expected_run_counts = {
            "train": 3,
            "val": 1,
            "test": 1,
        }

        if run_counts != expected_run_counts:
            raise RuntimeError(
                f"Incorrect split for {run_name}.\n"
                f"Expected: {expected_run_counts}\n"
                f"Found:    {run_counts}"
            )


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main() -> None:
    args = parse_arguments()
    configure_runtime_paths(args)

    print("=" * 76)
    print("10 µm Binary Segmentation — Coverage-Balanced Dataset Split")
    print("=" * 76)

    print("\nChecking source directories...")

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"Image root does not exist:\n"
            f"{IMAGE_ROOT}"
        )

    if not MASK_ROOT.exists():
        raise FileNotFoundError(
            f"Mask root does not exist:\n"
            f"{MASK_ROOT}"
        )

    print("Image root:")
    print(IMAGE_ROOT)

    print("\nMask root:")
    print(MASK_ROOT)

    all_run_pairs: dict[
        str,
        list[tuple[Path, Path]]
    ] = {}

    print("\nValidating manually labeled image-mask pairs...")

    for run_name in RUN_NAMES:
        image_folder = (
            IMAGE_ROOT
            / run_name
        )

        mask_folder = (
            MASK_ROOT
            / run_name
            / "masks"
        )

        pairs = pair_images_and_masks(
            image_folder=image_folder,
            mask_folder=mask_folder,
            run_name=run_name,
        )

        if len(pairs) != EXPECTED_PAIRS_PER_RUN:
            raise RuntimeError(
                f"{run_name} contains {len(pairs)} valid pairs, "
                f"but {EXPECTED_PAIRS_PER_RUN} were expected."
            )

        all_run_pairs[run_name] = pairs

        print(
            f"  {run_name}: {len(pairs)} labeled pairs passed"
        )

    samples_by_run = create_samples(
        all_run_pairs
    )

    selected_split, optimization_score = optimize_split(
        samples_by_run
    )

    verify_final_split(selected_split)

    print("\nBest coverage-balanced split found:")
    print(f"Optimization score: {optimization_score:.8f}")

    for split_name in (
        "train",
        "val",
        "test",
    ):
        values = [
            sample.coverage_percent
            for sample in selected_split[split_name]
        ]

        summary = summarize_coverage(values)

        print(
            f"  {split_name}: "
            f"mean={summary['mean']:.2f}%, "
            f"std={summary['std']:.2f}%, "
            f"min={summary['min']:.2f}%, "
            f"max={summary['max']:.2f}%"
        )

    print("\nCreating output directories...")

    output_directories = prepare_output_directories()

    records: list[dict[str, object]] = []

    print("\nCopying selected image-mask pairs...")

    for split_name in (
        "train",
        "val",
        "test",
    ):
        split_samples = sorted(
            selected_split[split_name],
            key=lambda sample: (
                int(
                    sample.run_name.replace(
                        "Run",
                        "",
                    )
                ),
                sample.image_path.name.lower(),
            ),
        )

        for sample in split_samples:
            (
                image_destination,
                mask_destination,
            ) = copy_sample(
                sample=sample,
                split_name=split_name,
                output_directories=output_directories,
            )

            records.append(
                {
                    "run": sample.run_name,
                    "split": split_name,
                    "pair_key": sample.pairing_key,
                    "coverage_percent": round(
                        sample.coverage_percent,
                        6,
                    ),
                    "source_image_name": sample.image_path.name,
                    "source_mask_name": sample.mask_path.name,
                    "output_image_name": image_destination.name,
                    "output_mask_name": mask_destination.name,
                }
            )

        print(
            f"  {split_name}: "
            f"{len(split_samples)} pairs copied"
        )

    split_order = {
        "train": 0,
        "val": 1,
        "test": 2,
    }

    records.sort(
        key=lambda record: (
            split_order[str(record["split"])],
            int(
                str(record["run"]).replace(
                    "Run",
                    "",
                )
            ),
            str(record["source_image_name"]).lower(),
        )
    )

    print("\nWriting reports...")

    write_summary_csv(records)

    write_statistics_txt(
        records=records,
        optimization_score=optimization_score,
    )

    print("\n" + "=" * 76)
    print("COVERAGE-BALANCED DATASET SPLIT COMPLETED SUCCESSFULLY")
    print("=" * 76)

    print("Train pairs:      27")
    print("Validation pairs: 9")
    print("Test pairs:       9")
    print("Total pairs:      45")

    print("\nOutput dataset:")
    print(OUTPUT_ROOT)

    print("\nCSV report:")
    print(SUMMARY_CSV)

    print("\nStatistics report:")
    print(STATISTICS_TXT)

    print("=" * 76)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\n" + "=" * 76)
        print("ERROR: COVERAGE-BALANCED SPLIT WAS NOT COMPLETED")
        print("=" * 76)
        print(error)
        print("=" * 76)

        sys.exit(1)
