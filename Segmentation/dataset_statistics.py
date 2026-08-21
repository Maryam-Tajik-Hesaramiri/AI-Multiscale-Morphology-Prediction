# ============================================================
# AI-driven Multiscale Morphology Prediction
#
# Phase 2.2 — 10 µm Binary Segmentation
#
# Script:
# dataset_statistics.py
#
# Purpose:
# Perform essential dataset quality control before segmentation
# model training.
#
# Main checks:
# - Count images and masks in Train / Validation / Test
# - Verify image-mask pairing
# - Verify image and mask dimensions match
# - Verify masks are binary
# - Calculate ZnO coverage percentage
# - Generate CSV and text summary reports
# - Generate a simple coverage histogram
#
# Classes:
#   0 = Al background
#   1 = ZnO
#
# Author: Maryam Tajik Hesaramiri
# Portfolio version: machine-specific paths removed.
# ============================================================
#
# Example:
#   python dataset_statistics.py --dataset-root /path/to/dataset_split_10um
#
# ============================================================

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = SCRIPT_ROOT / "dataset_split_10um"
OUTPUT_ROOT = SCRIPT_ROOT / "dataset_statistics_10um"

STATISTICS_CSV = OUTPUT_ROOT / "dataset_statistics.csv"
STATISTICS_TXT = OUTPUT_ROOT / "dataset_statistics.txt"
COVERAGE_HISTOGRAM = OUTPUT_ROOT / "coverage_histogram.png"

SPLIT_NAMES = ("train", "val", "test")

TIFF_EXTENSIONS = {".tif", ".tiff"}

EXPECTED_COUNTS = {
    "train": 27,
    "val": 9,
    "test": 9,
}

EXPECTED_TOTAL = 45

VALID_BINARY_VALUES = {
    frozenset({0}),
    frozenset({1}),
    frozenset({255}),
    frozenset({0, 1}),
    frozenset({0, 255}),
}


def parse_arguments() -> argparse.Namespace:
    """Parse dataset and output locations at runtime."""
    parser = argparse.ArgumentParser(
        description=(
            "Run quality control and descriptive statistics on a "
            "Train/Validation/Test SEM segmentation dataset."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "Dataset root containing train/, val/, and test/ folders, "
            "each with images/ and masks/ subfolders."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional output directory. "
            "Default: <script_dir>/dataset_statistics_10um"
        ),
    )
    return parser.parse_args()


def configure_runtime_paths(args: argparse.Namespace) -> None:
    """Apply runtime paths without embedding local machine information."""
    global DATASET_ROOT
    global OUTPUT_ROOT
    global STATISTICS_CSV
    global STATISTICS_TXT
    global COVERAGE_HISTOGRAM

    DATASET_ROOT = args.dataset_root.expanduser().resolve()

    if args.output_root is None:
        OUTPUT_ROOT = SCRIPT_ROOT / "dataset_statistics_10um"
    else:
        OUTPUT_ROOT = args.output_root.expanduser().resolve()

    STATISTICS_CSV = OUTPUT_ROOT / "dataset_statistics.csv"
    STATISTICS_TXT = OUTPUT_ROOT / "dataset_statistics.txt"
    COVERAGE_HISTOGRAM = OUTPUT_ROOT / "coverage_histogram.png"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_tiff_files(folder: Path) -> list[Path]:
    """
    Return TIFF files directly inside a folder.
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


def normalize_pairing_stem(path: Path) -> str:
    """
    Normalize image and mask names to a shared pairing key.

    Examples
    --------
    Image:
        Run1_5um-Image1_001.tif
        -> run1_5um-image1_001

    Mask:
        Run1_5um-Image1_001 - Labels.tif
        -> run1_5um-image1_001
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


def extract_run_name(filename: str) -> str:
    """
    Extract Run name from an output filename.

    Expected examples:
        Run1_5um-Image1_001.tif
        Run9_5um-Image9_010.tif
    """
    first_part = filename.split("_", maxsplit=1)[0]

    if not first_part.lower().startswith("run"):
        raise RuntimeError(
            f"Could not extract Run name from filename:\n{filename}"
        )

    run_number_text = first_part[3:]

    if not run_number_text.isdigit():
        raise RuntimeError(
            f"Invalid Run prefix in filename:\n{filename}"
        )

    return f"Run{int(run_number_text)}"


def build_unique_file_map(
    files: list[Path],
    file_type: str,
    split_name: str,
) -> dict[str, Path]:
    """
    Build a unique normalized-stem-to-file mapping.
    """
    file_map: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for file_path in files:
        pairing_key = normalize_pairing_stem(file_path)

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
            f"were found in split '{split_name}'."
        ]

        for key, paths in duplicates.items():
            lines.append(f"\nPairing key: {key}")

            for path in paths:
                lines.append(f"  - {path.name}")

        raise RuntimeError("\n".join(lines))

    return file_map


def pair_images_and_masks(
    image_folder: Path,
    mask_folder: Path,
    split_name: str,
) -> list[tuple[Path, Path]]:
    """
    Pair all images and masks within one split.
    """
    image_files = find_tiff_files(image_folder)
    mask_files = find_tiff_files(mask_folder)

    image_map = build_unique_file_map(
        image_files,
        file_type="image",
        split_name=split_name,
    )

    mask_map = build_unique_file_map(
        mask_files,
        file_type="mask",
        split_name=split_name,
    )

    image_keys = set(image_map)
    mask_keys = set(mask_map)

    images_without_masks = sorted(image_keys - mask_keys)
    masks_without_images = sorted(mask_keys - image_keys)

    if images_without_masks or masks_without_images:
        lines = [
            f"Image-mask pairing failed in split '{split_name}'."
        ]

        if images_without_masks:
            lines.append("\nImages without matching masks:")

            for key in images_without_masks:
                lines.append(f"  - {image_map[key].name}")

        if masks_without_images:
            lines.append("\nMasks without matching images:")

            for key in masks_without_images:
                lines.append(f"  - {mask_map[key].name}")

        raise RuntimeError("\n".join(lines))

    pairs = [
        (
            image_map[key],
            mask_map[key],
        )
        for key in sorted(image_keys)
    ]

    return pairs


def load_tiff(path: Path) -> np.ndarray:
    """
    Load a TIFF file as a NumPy array.
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
            f"Empty image array detected:\n{path}"
        )

    return array


def describe_array_shape(array: np.ndarray) -> tuple[int, int, int]:
    """
    Return height, width, and channel count.

    Grayscale:
        channels = 1

    RGB or multi-channel:
        channels = array.shape[2]
    """
    if array.ndim == 2:
        height, width = array.shape
        channels = 1

    elif array.ndim == 3:
        height, width, channels = array.shape

    else:
        raise RuntimeError(
            f"Unsupported array shape: {array.shape}"
        )

    return height, width, channels


def convert_mask_to_binary(
    mask_array: np.ndarray,
    mask_path: Path,
) -> tuple[np.ndarray, list[int], str]:
    """
    Validate and convert a mask to binary 0/1.

    Accepted masks:
        {0, 1}
        {0, 255}
        {0}
        {1}
        {255}
    """
    if mask_array.ndim == 3:
        # Accept RGB masks only when every channel is identical.
        first_channel = mask_array[..., 0]

        channels_are_identical = all(
            np.array_equal(
                first_channel,
                mask_array[..., channel_index],
            )
            for channel_index in range(
                1,
                mask_array.shape[2],
            )
        )

        if not channels_are_identical:
            raise RuntimeError(
                f"Mask has multiple non-identical channels:\n"
                f"{mask_path}\n"
                f"Shape: {mask_array.shape}"
            )

        mask_array = first_channel

    if mask_array.ndim != 2:
        raise RuntimeError(
            f"Mask must be 2D after loading:\n"
            f"{mask_path}\n"
            f"Shape: {mask_array.shape}"
        )

    unique_values = np.unique(mask_array)
    unique_value_set = frozenset(
        int(value)
        for value in unique_values
    )

    if unique_value_set not in VALID_BINARY_VALUES:
        raise RuntimeError(
            f"Mask is not binary:\n"
            f"{mask_path}\n"
            f"Unique values: {sorted(unique_value_set)}"
        )

    if unique_value_set.issubset({0, 1}):
        binary_mask = mask_array > 0
        encoding = "0/1"

    elif unique_value_set.issubset({0, 255}):
        binary_mask = mask_array > 0
        encoding = "0/255"

    else:
        raise RuntimeError(
            f"Unsupported binary mask encoding:\n"
            f"{mask_path}\n"
            f"Unique values: {sorted(unique_value_set)}"
        )

    return (
        binary_mask.astype(np.uint8),
        sorted(unique_value_set),
        encoding,
    )


def calculate_coverage(binary_mask: np.ndarray) -> dict[str, float | int]:
    """
    Calculate ZnO and background pixel statistics.
    """
    total_pixels = int(binary_mask.size)
    zno_pixels = int(np.count_nonzero(binary_mask))
    background_pixels = total_pixels - zno_pixels

    coverage_percent = (
        zno_pixels / total_pixels
    ) * 100.0

    background_percent = 100.0 - coverage_percent

    return {
        "total_pixels": total_pixels,
        "zno_pixels": zno_pixels,
        "background_pixels": background_pixels,
        "coverage_percent": coverage_percent,
        "background_percent": background_percent,
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    """
    Return descriptive statistics for a list of values.
    """
    if not values:
        raise ValueError(
            "Cannot summarize an empty value list."
        )

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "count": float(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1))
        if array.size > 1
        else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


# ============================================================
# REPORT FUNCTIONS
# ============================================================

def write_statistics_csv(
    records: list[dict[str, object]],
) -> None:
    """
    Write one row per image-mask pair.
    """
    fieldnames = [
        "split",
        "run",
        "pair_key",
        "image_name",
        "mask_name",
        "image_height",
        "image_width",
        "image_channels",
        "image_dtype",
        "mask_height",
        "mask_width",
        "mask_dtype",
        "mask_unique_values",
        "mask_encoding",
        "total_pixels",
        "zno_pixels",
        "background_pixels",
        "coverage_percent",
        "background_percent",
    ]

    with STATISTICS_CSV.open(
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
    split_summaries: dict[str, dict[str, float]],
    overall_summary: dict[str, float],
    warnings: list[str],
) -> None:
    """
    Write the human-readable dataset statistics report.
    """
    split_counts = Counter(
        str(record["split"])
        for record in records
    )

    run_counts = Counter(
        str(record["run"])
        for record in records
    )

    image_shapes = Counter(
        (
            int(record["image_height"]),
            int(record["image_width"]),
            int(record["image_channels"]),
        )
        for record in records
    )

    mask_shapes = Counter(
        (
            int(record["mask_height"]),
            int(record["mask_width"]),
        )
        for record in records
    )

    mask_encodings = Counter(
        str(record["mask_encoding"])
        for record in records
    )

    report_lines = [
        "=" * 76,
        "AI-driven Multiscale Morphology Prediction",
        "Phase 2.2 — 10 µm Binary Segmentation Dataset Statistics",
        "=" * 76,
        "",
        "Dataset and output paths are supplied at runtime.",
        "",
        "Dataset counts:",
        f"  Total image-mask pairs: {len(records)}",
        f"  Train:                 {split_counts['train']}",
        f"  Validation:            {split_counts['val']}",
        f"  Test:                  {split_counts['test']}",
        "",
        "Run representation:",
    ]

    for run_number in range(1, 10):
        run_name = f"Run{run_number}"

        report_lines.append(
            f"  {run_name}: {run_counts[run_name]} pairs"
        )

    report_lines.extend(
        [
            "",
            "Image dimensions and channels:",
        ]
    )

    for shape, count in sorted(image_shapes.items()):
        height, width, channels = shape

        report_lines.append(
            f"  {width} × {height}, channels={channels}: "
            f"{count} images"
        )

    report_lines.append("")
    report_lines.append("Mask dimensions:")

    for shape, count in sorted(mask_shapes.items()):
        height, width = shape

        report_lines.append(
            f"  {width} × {height}: {count} masks"
        )

    report_lines.append("")
    report_lines.append("Mask encoding:")

    for encoding, count in sorted(mask_encodings.items()):
        report_lines.append(
            f"  {encoding}: {count} masks"
        )

    report_lines.extend(
        [
            "",
            "ZnO coverage statistics:",
        ]
    )

    display_names = {
        "train": "Train",
        "val": "Validation",
        "test": "Test",
    }

    for split_name in SPLIT_NAMES:
        summary = split_summaries[split_name]

        report_lines.extend(
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

    report_lines.extend(
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
            "Quality-control results:",
            "- Image-mask pairing passed.",
            "- Image and mask dimensions matched for every pair.",
            "- All masks passed binary-value validation.",
            "- Expected split counts passed.",
            "- All 9 Runs were represented.",
        ]
    )

    if warnings:
        report_lines.extend(
            [
                "",
                "Warnings:",
            ]
        )

        for warning in warnings:
            report_lines.append(
                f"- {warning}"
            )

    else:
        report_lines.extend(
            [
                "",
                "Warnings:",
                "- None",
            ]
        )

    report_lines.extend(
        [
            "",
            "Generated files:",
            f"- {STATISTICS_CSV.name}",
            f"- {STATISTICS_TXT.name}",
            f"- {COVERAGE_HISTOGRAM.name}",
            "",
            "=" * 76,
        ]
    )

    STATISTICS_TXT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


def create_coverage_histogram(
    records: list[dict[str, object]],
) -> None:
    """
    Create a simple histogram of ZnO coverage by split.
    """
    coverage_by_split: dict[str, list[float]] = defaultdict(list)

    for record in records:
        split_name = str(record["split"])
        coverage = float(record["coverage_percent"])

        coverage_by_split[split_name].append(coverage)

    plt.figure(figsize=(9, 6))

    for split_name in SPLIT_NAMES:
        values = coverage_by_split[split_name]

        plt.hist(
            values,
            bins=10,
            alpha=0.5,
            label=split_name,
        )

    plt.xlabel("ZnO Coverage (%)")
    plt.ylabel("Number of Masks")
    plt.title("10 µm Binary Segmentation Dataset Coverage")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        COVERAGE_HISTOGRAM,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# FINAL VERIFICATION
# ============================================================

def verify_final_dataset(
    records: list[dict[str, object]],
) -> None:
    """
    Verify expected dataset counts and Run representation.
    """
    if len(records) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} total pairs, "
            f"but found {len(records)}."
        )

    split_counts = Counter(
        str(record["split"])
        for record in records
    )

    for split_name, expected_count in EXPECTED_COUNTS.items():
        actual_count = split_counts[split_name]

        if actual_count != expected_count:
            raise RuntimeError(
                f"Incorrect count for split '{split_name}'.\n"
                f"Expected: {expected_count}\n"
                f"Found:    {actual_count}"
            )

    run_counts = Counter(
        str(record["run"])
        for record in records
    )

    expected_run_names = {
        f"Run{run_number}"
        for run_number in range(1, 10)
    }

    actual_run_names = set(run_counts)

    if actual_run_names != expected_run_names:
        missing_runs = sorted(
            expected_run_names - actual_run_names
        )

        unexpected_runs = sorted(
            actual_run_names - expected_run_names
        )

        raise RuntimeError(
            "Run representation check failed.\n"
            f"Missing Runs: {missing_runs}\n"
            f"Unexpected Runs: {unexpected_runs}"
        )

    for run_name in sorted(expected_run_names):
        if run_counts[run_name] != 5:
            raise RuntimeError(
                f"{run_name} should contain 5 total pairs, "
                f"but contains {run_counts[run_name]}."
            )


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main() -> None:
    """
    Run complete essential dataset statistics and quality control.
    """
    args = parse_arguments()
    configure_runtime_paths(args)

    print("=" * 76)
    print("10 µm Binary Segmentation — Dataset Statistics")
    print("=" * 76)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset split folder was not found:\n"
            f"{DATASET_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    records: list[dict[str, object]] = []
    warnings: list[str] = []

    print("\nChecking dataset splits...")

    for split_name in SPLIT_NAMES:
        image_folder = (
            DATASET_ROOT
            / split_name
            / "images"
        )

        mask_folder = (
            DATASET_ROOT
            / split_name
            / "masks"
        )

        print(f"\nProcessing split: {split_name}")

        pairs = pair_images_and_masks(
            image_folder=image_folder,
            mask_folder=mask_folder,
            split_name=split_name,
        )

        expected_count = EXPECTED_COUNTS[split_name]

        if len(pairs) != expected_count:
            raise RuntimeError(
                f"Split '{split_name}' contains "
                f"{len(pairs)} pairs, but "
                f"{expected_count} were expected."
            )

        print(
            f"  {len(pairs)} image-mask pairs found"
        )

        for pair_index, (
            image_path,
            mask_path,
        ) in enumerate(
            pairs,
            start=1,
        ):
            image_array = load_tiff(image_path)
            mask_array = load_tiff(mask_path)

            (
                image_height,
                image_width,
                image_channels,
            ) = describe_array_shape(image_array)

            (
                mask_height_original,
                mask_width_original,
                _,
            ) = describe_array_shape(mask_array)

            (
                binary_mask,
                mask_unique_values,
                mask_encoding,
            ) = convert_mask_to_binary(
                mask_array=mask_array,
                mask_path=mask_path,
            )

            mask_height, mask_width = binary_mask.shape

            if (
                image_height != mask_height
                or image_width != mask_width
            ):
                raise RuntimeError(
                    f"Image-mask dimension mismatch:\n"
                    f"Image: {image_path}\n"
                    f"Image dimensions: "
                    f"{image_width} × {image_height}\n\n"
                    f"Mask: {mask_path}\n"
                    f"Mask dimensions: "
                    f"{mask_width} × {mask_height}"
                )

            coverage_results = calculate_coverage(
                binary_mask
            )

            if coverage_results["coverage_percent"] == 0:
                warnings.append(
                    f"Mask contains no ZnO pixels: "
                    f"{mask_path.name}"
                )

            if coverage_results["coverage_percent"] == 100:
                warnings.append(
                    f"Mask contains no Al background pixels: "
                    f"{mask_path.name}"
                )

            run_name = extract_run_name(
                image_path.name
            )

            record = {
                "split": split_name,
                "run": run_name,
                "pair_key": normalize_pairing_stem(
                    image_path
                ),
                "image_name": image_path.name,
                "mask_name": mask_path.name,
                "image_height": image_height,
                "image_width": image_width,
                "image_channels": image_channels,
                "image_dtype": str(image_array.dtype),
                "mask_height": mask_height,
                "mask_width": mask_width,
                "mask_dtype": str(mask_array.dtype),
                "mask_unique_values": ";".join(
                    str(value)
                    for value in mask_unique_values
                ),
                "mask_encoding": mask_encoding,
                "total_pixels": coverage_results[
                    "total_pixels"
                ],
                "zno_pixels": coverage_results[
                    "zno_pixels"
                ],
                "background_pixels": coverage_results[
                    "background_pixels"
                ],
                "coverage_percent": round(
                    float(
                        coverage_results[
                            "coverage_percent"
                        ]
                    ),
                    6,
                ),
                "background_percent": round(
                    float(
                        coverage_results[
                            "background_percent"
                        ]
                    ),
                    6,
                ),
            }

            records.append(record)

            print(
                f"  [{pair_index:02d}/{len(pairs):02d}] "
                f"{image_path.name} — "
                f"Coverage: "
                f"{coverage_results['coverage_percent']:.2f}%"
            )

    print("\nVerifying complete dataset...")

    verify_final_dataset(records)

    coverage_by_split: dict[str, list[float]] = defaultdict(list)

    for record in records:
        coverage_by_split[
            str(record["split"])
        ].append(
            float(record["coverage_percent"])
        )

    split_summaries = {
        split_name: summarize_values(
            coverage_by_split[split_name]
        )
        for split_name in SPLIT_NAMES
    }

    all_coverage_values = [
        float(record["coverage_percent"])
        for record in records
    ]

    overall_summary = summarize_values(
        all_coverage_values
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
            str(record["image_name"]).lower(),
        )
    )

    print("Writing CSV report...")

    write_statistics_csv(records)

    print("Creating coverage histogram...")

    create_coverage_histogram(records)

    print("Writing text report...")

    write_statistics_txt(
        records=records,
        split_summaries=split_summaries,
        overall_summary=overall_summary,
        warnings=warnings,
    )

    print("\n" + "=" * 76)
    print("DATASET STATISTICS COMPLETED SUCCESSFULLY")
    print("=" * 76)

    print(f"Total pairs:      {len(records)}")
    print(f"Train pairs:      {EXPECTED_COUNTS['train']}")
    print(f"Validation pairs: {EXPECTED_COUNTS['val']}")
    print(f"Test pairs:       {EXPECTED_COUNTS['test']}")

    print("\nCoverage summary:")

    for split_name in SPLIT_NAMES:
        summary = split_summaries[split_name]

        print(
            f"  {split_name}: "
            f"mean={summary['mean']:.2f}%, "
            f"std={summary['std']:.2f}%, "
            f"min={summary['min']:.2f}%, "
            f"max={summary['max']:.2f}%"
        )

    print(
        f"  overall: "
        f"mean={overall_summary['mean']:.2f}%, "
        f"std={overall_summary['std']:.2f}%, "
        f"min={overall_summary['min']:.2f}%, "
        f"max={overall_summary['max']:.2f}%"
    )

    if warnings:
        print(f"\nWarnings: {len(warnings)}")

        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nWarnings: None")

    print("\nGenerated outputs:")
    print(STATISTICS_CSV)
    print(STATISTICS_TXT)
    print(COVERAGE_HISTOGRAM)

    print("=" * 76)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\n" + "=" * 76)
        print("ERROR: DATASET STATISTICS WAS NOT COMPLETED")
        print("=" * 76)
        print(error)
        print("=" * 76)

        sys.exit(1)
