#!/usr/bin/env python
"""
build_ai_dataset.py
===================

Build the final morphology-prediction dataset by combining:

1. DOE / Taguchi growth parameters
2. U-Net predicted ZnO coverage
3. SEM-derived nanosheet width measurements

The resulting datasets are:

- Image-level dataset:
    One row per SEM image.
    Retains within-Run morphology variability.

- Run-level dataset:
    One row per independent DOE condition.
    Recommended for synthesis-parameter-to-morphology modeling.

Important
---------
The SEM images are repeated measurements nested within a small number of
independent experimental Runs. They must not be treated as independent DOE
conditions during model validation.

Use Run-grouped validation or Leave-One-Run-Out (LORO) for downstream
morphology prediction.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DOE_PATH = (
    SCRIPT_DIR.parent
    / "data"
    / "Taguchi_design.xlsx"
)

DEFAULT_UNET_PATH = (
    SCRIPT_DIR.parent
    / "segmentation"
    / "full_dataset_predictions"
    / "inference_manifest.csv"
)

DEFAULT_WIDTH_PATH = (
    SCRIPT_DIR.parent
    / "data"
    / "width_measurements.csv"
)

DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR
    / "ai_dataset"
)


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset construction settings."""

    expected_runs: int = 9
    expected_images: int = 458
    figure_dpi: int = 300

    input_columns: Tuple[str, ...] = (
        "rpm",
        "ph",
        "time_h",
        "temperature_c",
    )

    target_columns: Tuple[str, ...] = (
        "zno_coverage_percent",
        "mean_width_um",
    )

    def validate(self) -> None:
        if self.expected_runs <= 0:
            raise ValueError(
                "expected_runs must be positive."
            )

        if self.expected_images <= 0:
            raise ValueError(
                "expected_images must be positive."
            )

        if self.figure_dpi <= 0:
            raise ValueError(
                "figure_dpi must be positive."
            )


# =============================================================================
# IDENTIFIERS
# =============================================================================

RUN_PATTERN = re.compile(
    r"run[\s_-]*(\d+)",
    re.IGNORECASE,
)


def normalize_run(value: object) -> str:
    """
    Convert Run identifiers into a consistent format.

    Examples
    --------
    1       -> Run1
    "1"     -> Run1
    "run_1" -> Run1
    "Run1"  -> Run1
    """

    text = str(value).strip()

    match = RUN_PATTERN.search(text)

    if match:
        return f"Run{int(match.group(1))}"

    try:
        return f"Run{int(float(text))}"

    except ValueError as exc:
        raise ValueError(
            f"Cannot interpret Run identifier: {value}"
        ) from exc


def run_number(run_name: str) -> int:
    """Extract the integer Run number."""

    match = RUN_PATTERN.search(
        str(run_name)
    )

    if not match:
        raise ValueError(
            f"Invalid Run name: {run_name}"
        )

    return int(
        match.group(1)
    )


def canonical_sample_id(value: str) -> str:
    """
    Normalize image identity across files and extensions.

    Example
    -------
    5um-Image1_001.tif -> 5um-image1_001
    """

    value = str(value).strip()

    value = re.sub(
        r"\.[^.]+$",
        "",
        value,
    )

    return value.lower()


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def ensure_numeric(
    dataframe: pd.DataFrame,
    columns: List[str],
    table_name: str,
) -> None:
    """Convert required columns to numeric."""

    for column in columns:
        converted = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        bad = (
            converted.isna()
            & dataframe[column].notna()
        )

        if bad.any():
            examples = (
                dataframe.loc[
                    bad,
                    column,
                ]
                .head(5)
                .tolist()
            )

            raise ValueError(
                f"{table_name}: non-numeric values "
                f"in '{column}': {examples}"
            )

        dataframe[column] = converted


# =============================================================================
# DOE DATA
# =============================================================================

def load_doe(
    path: Path,
    config: DatasetConfig,
) -> pd.DataFrame:
    """Load and standardize DOE growth parameters."""

    if not path.is_file():
        raise FileNotFoundError(
            f"DOE file not found:\n{path}"
        )

    doe = pd.read_excel(
        path
    )

    lookup = {
        str(column).strip().lower(): column
        for column in doe.columns
    }

    aliases = {
        "test": [
            "test",
            "run",
        ],
        "rpm": [
            "rpm",
        ],
        "ph": [
            "ph",
        ],
        "time_h": [
            "time_h",
            "time",
            "time (h)",
        ],
        "temperature_c": [
            "temperature_c",
            "tempreture_c",
            "temperature",
            "temp_c",
        ],
    }

    selected: Dict[str, str] = {}

    for standard_name, options in aliases.items():
        found = None

        for option in options:
            if option.lower() in lookup:
                found = lookup[
                    option.lower()
                ]
                break

        if found is None:
            raise KeyError(
                f"DOE column '{standard_name}' "
                f"not found.\n"
                f"Available columns: "
                f"{list(doe.columns)}"
            )

        selected[
            standard_name
        ] = found

    doe = doe[
        [
            selected["test"],
            selected["rpm"],
            selected["ph"],
            selected["time_h"],
            selected["temperature_c"],
        ]
    ].copy()

    doe.columns = [
        "test",
        "rpm",
        "ph",
        "time_h",
        "temperature_c",
    ]

    ensure_numeric(
        doe,
        [
            "test",
            "rpm",
            "ph",
            "time_h",
            "temperature_c",
        ],
        "DOE",
    )

    doe["run"] = doe[
        "test"
    ].map(
        normalize_run
    )

    if doe["run"].duplicated().any():
        raise RuntimeError(
            "DOE contains duplicate Run definitions."
        )

    if len(doe) != config.expected_runs:
        raise RuntimeError(
            f"Expected {config.expected_runs} Runs, "
            f"found {len(doe)}."
        )

    expected_runs = {
        f"Run{i}"
        for i in range(
            1,
            config.expected_runs + 1,
        )
    }

    actual_runs = set(
        doe["run"]
    )

    if actual_runs != expected_runs:
        raise RuntimeError(
            "DOE Run identities do not match "
            "the expected experimental design."
        )

    doe["run_number"] = (
        doe["run"]
        .map(run_number)
    )

    return (
        doe[
            [
                "run",
                "run_number",
                "rpm",
                "ph",
                "time_h",
                "temperature_c",
            ]
        ]
        .sort_values(
            "run_number"
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# U-NET COVERAGE
# =============================================================================

def load_unet_predictions(
    path: Path,
    config: DatasetConfig,
) -> pd.DataFrame:
    """Load full-dataset U-Net coverage predictions."""

    if not path.is_file():
        raise FileNotFoundError(
            f"U-Net inference manifest not found:\n{path}"
        )

    dataframe = pd.read_csv(
        path
    )

    required = {
        "run",
        "sample_id",
        "predicted_coverage_percent",
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise KeyError(
            "U-Net manifest is missing columns: "
            f"{sorted(missing)}"
        )

    if len(dataframe) != config.expected_images:
        raise RuntimeError(
            f"Expected {config.expected_images} "
            f"U-Net rows, found {len(dataframe)}."
        )

    dataframe = (
        dataframe.copy()
    )

    dataframe["run"] = (
        dataframe["run"]
        .map(normalize_run)
    )

    dataframe["sample_key"] = (
        dataframe["sample_id"]
        .map(canonical_sample_id)
    )

    dataframe[
        "zno_coverage_percent"
    ] = pd.to_numeric(
        dataframe[
            "predicted_coverage_percent"
        ],
        errors="coerce",
    )

    coverage = dataframe[
        "zno_coverage_percent"
    ]

    if coverage.isna().any():
        raise ValueError(
            "U-Net coverage contains "
            "missing or non-numeric values."
        )

    if (
        (coverage < 0)
        | (coverage > 100)
    ).any():
        raise ValueError(
            "U-Net coverage must be "
            "between 0 and 100%."
        )

    if dataframe.duplicated(
        [
            "run",
            "sample_key",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate image identities "
            "found in U-Net predictions."
        )

    keep = [
        "run",
        "sample_key",
        "sample_id",
        "zno_coverage_percent",
    ]

    optional = [
        "source_image",
        "predicted_mask_path",
        "probability_map_path",
        "mean_zno_probability",
    ]

    for column in optional:
        if column in dataframe.columns:
            keep.append(
                column
            )

    return dataframe[
        keep
    ].copy()


# =============================================================================
# WIDTH DATA
# =============================================================================

def load_width_measurements(
    path: Path,
    config: DatasetConfig,
) -> pd.DataFrame:
    """Load SEM nanosheet width measurements."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Width dataset not found:\n{path}"
        )

    dataframe = pd.read_csv(
        path
    )

    required = {
        "run",
        "image_name",
        "mean_width_um",
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise KeyError(
            "Width dataset is missing columns: "
            f"{sorted(missing)}"
        )

    if len(dataframe) != config.expected_images:
        raise RuntimeError(
            f"Expected {config.expected_images} "
            f"width rows, found {len(dataframe)}."
        )

    dataframe = (
        dataframe.copy()
    )

    dataframe["run"] = (
        dataframe["run"]
        .map(normalize_run)
    )

    dataframe["sample_key"] = (
        dataframe["image_name"]
        .map(canonical_sample_id)
    )

    numeric_columns = [
        column
        for column in [
            "mean_width_um",
            "median_width_um",
            "width_std_um",
            "width_p10_um",
            "width_p90_um",
            "valid_width_count",
            "valid_profile_fraction",
        ]
        if column in dataframe.columns
    ]

    ensure_numeric(
        dataframe,
        numeric_columns,
        "Width dataset",
    )

    if dataframe[
        "mean_width_um"
    ].isna().any():
        raise ValueError(
            "mean_width_um contains missing values."
        )

    if (
        dataframe[
            "mean_width_um"
        ] <= 0
    ).any():
        raise ValueError(
            "mean_width_um contains "
            "non-positive values."
        )

    if dataframe.duplicated(
        [
            "run",
            "sample_key",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate image identities "
            "found in width measurements."
        )

    keep = [
        "run",
        "sample_key",
        "image_name",
        "mean_width_um",
    ]

    optional = [
        "median_width_um",
        "width_std_um",
        "width_p10_um",
        "width_p90_um",
        "valid_width_count",
        "valid_profile_fraction",
    ]

    for column in optional:
        if column in dataframe.columns:
            keep.append(
                column
            )

    dataframe = dataframe[
        keep
    ].copy()

    dataframe = dataframe.rename(
        columns={
            "width_std_um":
                "within_image_width_std_um",
            "valid_width_count":
                "width_valid_measurement_count",
            "valid_profile_fraction":
                "width_valid_profile_fraction",
        }
    )

    return dataframe


# =============================================================================
# IMAGE-LEVEL DATASET
# =============================================================================

def build_image_level_dataset(
    doe: pd.DataFrame,
    unet: pd.DataFrame,
    width: pd.DataFrame,
    config: DatasetConfig,
) -> pd.DataFrame:
    """Merge morphology measurements image-by-image."""

    merged = unet.merge(
        width,
        on=[
            "run",
            "sample_key",
        ],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = (
        merged["_merge"]
        != "both"
    )

    if unmatched.any():
        raise RuntimeError(
            "U-Net and width datasets do not "
            "match exactly image-by-image."
        )

    merged = merged.drop(
        columns="_merge"
    )

    if len(merged) != config.expected_images:
        raise RuntimeError(
            f"Expected {config.expected_images} "
            f"matched images, found {len(merged)}."
        )

    final = merged.merge(
        doe,
        on="run",
        how="left",
        validate="many_to_one",
    )

    required_doe = [
        "run_number",
        "rpm",
        "ph",
        "time_h",
        "temperature_c",
    ]

    if final[
        required_doe
    ].isna().any().any():
        raise RuntimeError(
            "Some images could not be assigned "
            "DOE parameters."
        )

    final["image"] = (
        final["sample_id"]
    )

    for run_name, group in final.groupby(
        "run"
    ):
        unique_conditions = group[
            [
                "rpm",
                "ph",
                "time_h",
                "temperature_c",
            ]
        ].drop_duplicates()

        if len(unique_conditions) != 1:
            raise RuntimeError(
                f"{run_name} maps to more than "
                "one DOE condition."
            )

    final = (
        final.sort_values(
            [
                "run_number",
                "sample_key",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    primary_columns = [
        "run",
        "run_number",
        "image",
        "rpm",
        "ph",
        "time_h",
        "temperature_c",
        "zno_coverage_percent",
        "mean_width_um",
    ]

    remaining = [
        column
        for column in final.columns
        if column not in primary_columns
    ]

    return final[
        primary_columns
        + remaining
    ]


# =============================================================================
# RUN-LEVEL DATASET
# =============================================================================

def build_run_level_dataset(
    image_level: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate repeated SEM observations into independent DOE Runs.
    """

    rows = []

    for run_name, group in image_level.groupby(
        "run",
        sort=False,
    ):
        coverage = group[
            "zno_coverage_percent"
        ].astype(float)

        width = group[
            "mean_width_um"
        ].astype(float)

        rows.append(
            {
                "run": run_name,
                "run_number": int(
                    group[
                        "run_number"
                    ].iloc[0]
                ),
                "rpm": float(
                    group[
                        "rpm"
                    ].iloc[0]
                ),
                "ph": float(
                    group[
                        "ph"
                    ].iloc[0]
                ),
                "time_h": float(
                    group[
                        "time_h"
                    ].iloc[0]
                ),
                "temperature_c": float(
                    group[
                        "temperature_c"
                    ].iloc[0]
                ),
                "n_images": int(
                    len(group)
                ),

                "zno_coverage_percent_mean":
                    float(
                        coverage.mean()
                    ),

                "zno_coverage_percent_sd":
                    float(
                        coverage.std(
                            ddof=1
                        )
                    ),

                "zno_coverage_percent_median":
                    float(
                        coverage.median()
                    ),

                "zno_coverage_percent_min":
                    float(
                        coverage.min()
                    ),

                "zno_coverage_percent_max":
                    float(
                        coverage.max()
                    ),

                "mean_width_um_mean":
                    float(
                        width.mean()
                    ),

                "mean_width_um_sd_across_images":
                    float(
                        width.std(
                            ddof=1
                        )
                    ),

                "mean_width_um_median_across_images":
                    float(
                        width.median()
                    ),

                "mean_width_um_min":
                    float(
                        width.min()
                    ),

                "mean_width_um_max":
                    float(
                        width.max()
                    ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "run_number"
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# FINAL VALIDATION
# =============================================================================

def validate_datasets(
    image_level: pd.DataFrame,
    run_level: pd.DataFrame,
    config: DatasetConfig,
) -> None:
    """Validate final dataset structure."""

    if len(image_level) != config.expected_images:
        raise RuntimeError(
            "Unexpected number of image-level rows."
        )

    if len(run_level) != config.expected_runs:
        raise RuntimeError(
            "Unexpected number of Run-level rows."
        )

    required = (
        list(
            config.input_columns
        )
        + list(
            config.target_columns
        )
    )

    if image_level[
        required
    ].isna().any().any():
        raise RuntimeError(
            "Primary AI dataset contains "
            "missing values."
        )

    if image_level.duplicated(
        [
            "run",
            "image",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate image rows found."
        )

    unique_conditions = (
        image_level[
            list(
                config.input_columns
            )
        ]
        .drop_duplicates()
    )

    if len(
        unique_conditions
    ) != config.expected_runs:
        raise RuntimeError(
            "Number of unique DOE conditions "
            "does not match expected Runs."
        )


# =============================================================================
# FIGURES
# =============================================================================

def save_figures(
    image_level: pd.DataFrame,
    run_level: pd.DataFrame,
    output_dir: Path,
    config: DatasetConfig,
) -> None:
    """Create basic morphology QC figures."""

    figure_dir = (
        output_dir
        / "figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_order = [
        f"Run{i}"
        for i in range(
            1,
            config.expected_runs + 1,
        )
    ]

    # Coverage by Run
    coverage_groups = [
        image_level.loc[
            image_level["run"] == run,
            "zno_coverage_percent",
        ].to_numpy()
        for run in run_order
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.boxplot(
        coverage_groups,
        tick_labels=run_order,
    )

    ax.set_xlabel(
        "Experimental Run"
    )

    ax.set_ylabel(
        "Predicted ZnO coverage (%)"
    )

    ax.set_title(
        "ZnO coverage distribution by Run"
    )

    fig.tight_layout()

    fig.savefig(
        figure_dir
        / "coverage_by_run.png",
        dpi=config.figure_dpi,
    )

    plt.close(
        fig
    )

    # Width by Run
    width_groups = [
        image_level.loc[
            image_level["run"] == run,
            "mean_width_um",
        ].to_numpy()
        for run in run_order
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.boxplot(
        width_groups,
        tick_labels=run_order,
    )

    ax.set_xlabel(
        "Experimental Run"
    )

    ax.set_ylabel(
        "Mean nanosheet width (µm)"
    )

    ax.set_title(
        "Nanosheet width distribution by Run"
    )

    fig.tight_layout()

    fig.savefig(
        figure_dir
        / "width_by_run.png",
        dpi=config.figure_dpi,
    )

    plt.close(
        fig
    )

    # Coverage vs Width
    fig, ax = plt.subplots(
        figsize=(6.5, 5.5)
    )

    ax.scatter(
        image_level[
            "zno_coverage_percent"
        ],
        image_level[
            "mean_width_um"
        ],
        alpha=0.65,
    )

    ax.set_xlabel(
        "Predicted ZnO coverage (%)"
    )

    ax.set_ylabel(
        "Mean nanosheet width (µm)"
    )

    ax.set_title(
        "Image-level morphology targets"
    )

    fig.tight_layout()

    fig.savefig(
        figure_dir
        / "coverage_vs_width.png",
        dpi=config.figure_dpi,
    )

    plt.close(
        fig
    )


# =============================================================================
# REPORTING
# =============================================================================

def write_summary(
    run_level: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write a compact dataset summary."""

    lines = [
        "=" * 72,
        "AI MORPHOLOGY DATASET SUMMARY",
        "=" * 72,
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Independent DOE Runs: {len(run_level)}",
        f"Total SEM images:      {int(run_level['n_images'].sum())}",
        "",
        "Inputs:",
        "  RPM",
        "  pH",
        "  Time (h)",
        "  Temperature (°C)",
        "",
        "Targets:",
        "  ZnO coverage (%)",
        "  Mean nanosheet width (µm)",
        "",
        "Important modeling rule:",
        "SEM images are repeated observations nested within DOE Runs.",
        "Use Run-grouped validation or Leave-One-Run-Out.",
        "",
        "Run-level summary:",
        "-" * 72,
    ]

    for _, row in run_level.iterrows():
        lines.append(
            f"{row['run']:<6} | "
            f"Coverage="
            f"{row['zno_coverage_percent_mean']:.3f} "
            f"± {row['zno_coverage_percent_sd']:.3f}% | "
            f"Width="
            f"{row['mean_width_um_mean']:.4f} "
            f"± "
            f"{row['mean_width_um_sd_across_images']:.4f} µm | "
            f"n={int(row['n_images'])}"
        )

    (
        output_dir
        / "dataset_summary.txt"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


def write_metadata(
    config: DatasetConfig,
    doe_path: Path,
    unet_path: Path,
    width_path: Path,
    output_dir: Path,
) -> None:
    """Save reproducibility metadata."""

    metadata = {
        **asdict(config),
        "doe_path": str(
            doe_path
        ),
        "unet_manifest_path": str(
            unet_path
        ),
        "width_dataset_path": str(
            width_path
        ),
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "validation_note": (
            "Repeated SEM images are nested "
            "within independent DOE Runs. "
            "Use grouped or LORO validation."
        ),
    }

    (
        output_dir
        / "dataset_config.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )


# =============================================================================
# PIPELINE
# =============================================================================

def build_dataset(
    doe_path: Path,
    unet_path: Path,
    width_path: Path,
    output_dir: Path,
) -> None:
    """Build image-level and Run-level morphology datasets."""

    config = DatasetConfig()
    config.validate()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("BUILD AI MORPHOLOGY DATASET")
    print("=" * 72)

    print("Loading DOE parameters...")
    doe = load_doe(
        doe_path,
        config,
    )

    print("Loading U-Net predictions...")
    unet = load_unet_predictions(
        unet_path,
        config,
    )

    print("Loading width measurements...")
    width = load_width_measurements(
        width_path,
        config,
    )

    print("Matching image-level data...")
    image_level = (
        build_image_level_dataset(
            doe=doe,
            unet=unet,
            width=width,
            config=config,
        )
    )

    print("Building Run-level dataset...")
    run_level = (
        build_run_level_dataset(
            image_level
        )
    )

    validate_datasets(
        image_level,
        run_level,
        config,
    )

    image_path = (
        output_dir
        / "image_level.csv"
    )

    run_path = (
        output_dir
        / "run_level.csv"
    )

    image_level.to_csv(
        image_path,
        index=False,
    )

    run_level.to_csv(
        run_path,
        index=False,
    )

    save_figures(
        image_level=image_level,
        run_level=run_level,
        output_dir=output_dir,
        config=config,
    )

    write_summary(
        run_level,
        output_dir,
    )

    write_metadata(
        config=config,
        doe_path=doe_path,
        unet_path=unet_path,
        width_path=width_path,
        output_dir=output_dir,
    )

    print("\n" + "=" * 72)
    print("DATASET BUILD COMPLETE")
    print("=" * 72)

    print(
        f"Image-level rows : {len(image_level)}"
    )

    print(
        f"Independent Runs : {len(run_level)}"
    )

    print(
        f"\nImage dataset: {image_path}"
    )

    print(
        f"Run dataset:   {run_path}"
    )

    print(
        "\nImportant: use Run-grouped / "
        "Leave-One-Run-Out validation."
    )


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build morphology-prediction datasets "
            "from DOE parameters, U-Net coverage, "
            "and SEM width measurements."
        )
    )

    parser.add_argument(
        "--doe",
        type=Path,
        default=DEFAULT_DOE_PATH,
        help="Path to DOE / Taguchi design file.",
    )

    parser.add_argument(
        "--unet",
        type=Path,
        default=DEFAULT_UNET_PATH,
        help="Path to U-Net inference manifest.",
    )

    parser.add_argument(
        "--width",
        type=Path,
        default=DEFAULT_WIDTH_PATH,
        help="Path to width-measurement CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated AI datasets.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    build_dataset(
        doe_path=args.doe.expanduser().resolve(),
        unet_path=args.unet.expanduser().resolve(),
        width_path=args.width.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
