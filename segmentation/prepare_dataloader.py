#!/usr/bin/env python
"""
prepare_dataloader.py
===========================

Run-balanced + coverage-aware DataLoader
for binary ZnO segmentation.

WHY THIS SAMPLER?
-----------------
Audit of the previous training sampler showed:

- Full training-image mean foreground: ~80.61%
- Sampled-patch mean foreground:       ~85.88%
- Sampled-patch median foreground:     100%
- >=99% foreground patches:            ~70.30%

A first coverage-balanced attempt corrected this foreground bias,
but overrepresented sparse Runs such as Run1 and Run6.

This sampler therefore balances TWO things:

1. RUN REPRESENTATION
   Every experimental Run contributes the same number of training
   patches per epoch.

2. WITHIN-RUN COVERAGE DIVERSITY
   Within each Run, patches are sampled across the foreground-coverage
   categories that actually exist in that Run.

For 27 training images, 32 patches/image/epoch:
    27 * 32 = 864 training patches/epoch

For 9 Runs:
    864 / 9 = 96 patches per Run per epoch

Unchanged:
- Original Train/Validation/Test split
- Image-mask pairing
- SEM normalization
- Binary mask preparation
- Patch size = 256 x 256
- Validation/Test sliding windows
- Validation/Test stride = 128
- Training augmentation
- Random seed = 42
- Batch sizes
- Original dataset files are never modified.

Portfolio version:
- Uses relative/runtime-configurable dataset paths
- Preserves the research sampling logic
- Avoids machine-specific paths or unpublished data
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tifffile
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# 1. PROJECT CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset_split_10um"

SEED = 42

PATCH_SIZE = (256, 256)
TRAIN_PATCHES_PER_IMAGE = 32

VAL_STRIDE = (128, 128)
TEST_STRIDE = (128, 128)

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

NUM_WORKERS = 0

# Candidate locations are searched more densely than Val/Test inference.
CANDIDATE_STRIDE = (64, 64)

# Robust SEM normalization.
LOW_PERCENTILE = 0.5
HIGH_PERCENTILE = 99.5

SUPPORTED_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}

MASK_SUFFIXES = (
    "_mask",
    "-mask",
    " - mask",
    "_labels",
    "-labels",
    " - labels",
    "_seg",
    "-seg",
    " - seg",
)

CATEGORY_NAMES = (
    "low_0_25",
    "mixed_25_75",
    "high_75_99",
    "almost_all_99_100",
)


def set_dataset_dir(path: Path) -> None:
    """Update the dataset root at runtime."""
    global DATASET_DIR
    DATASET_DIR = Path(path).expanduser().resolve()


# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class ImageMaskPair:
    image_path: Path
    mask_path: Path
    sample_id: str


@dataclass(frozen=True)
class PatchLocation:
    pair_index: int
    top: int
    left: int
    original_height: int
    original_width: int


@dataclass(frozen=True)
class PatchCandidate:
    pair_index: int
    top: int
    left: int
    foreground_fraction: float


# =============================================================================
# 3. REPRODUCIBILITY
# =============================================================================

def seed_everything(seed: int = SEED) -> None:

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


def seed_worker(worker_id: int) -> None:

    worker_seed = (
        torch.initial_seed() % (2**32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# 4. RUN IDENTIFICATION
# =============================================================================

def extract_run(sample_id: str) -> str:
    """
    Extract Run number from names such as:
        Run5_5um-Image1_008
    """

    match = re.search(
        r"run\s*[_-]?(\d+)",
        sample_id,
        flags=re.IGNORECASE,
    )

    if match:
        return f"Run{int(match.group(1))}"

    raise ValueError(
        f"Could not determine Run from sample ID: {sample_id}"
    )


def run_sort_key(run_name: str) -> int:

    match = re.search(r"\d+", run_name)

    if match:
        return int(match.group())

    return 999


# =============================================================================
# 5. FILE DISCOVERY / PAIRING
# =============================================================================

def list_supported_files(
    folder: Path,
) -> List[Path]:

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Required folder not found: {folder}"
        )

    files = sorted(
        [
            path
            for path in folder.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ],
        key=lambda path: path.name.lower(),
    )

    if not files:
        raise RuntimeError(
            f"No supported image files found in: {folder}"
        )

    return files


def canonical_mask_stem(
    stem: str,
) -> str:

    cleaned = stem.strip()
    cleaned_lower = cleaned.lower()

    for suffix in MASK_SUFFIXES:

        if cleaned_lower.endswith(suffix):

            cleaned = cleaned[
                : -len(suffix)
            ].rstrip(" _-")

            break

    return cleaned


def discover_pairs(
    split_name: str,
) -> List[ImageMaskPair]:

    split_dir = DATASET_DIR / split_name

    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"

    image_files = list_supported_files(
        image_dir
    )

    mask_files = list_supported_files(
        mask_dir
    )

    image_map: Dict[str, Path] = {}
    mask_map: Dict[str, Path] = {}

    for image_path in image_files:

        key = image_path.stem.lower()

        if key in image_map:
            raise RuntimeError(
                f"Duplicate image stem: "
                f"{image_path.stem}"
            )

        image_map[key] = image_path

    for mask_path in mask_files:

        key = canonical_mask_stem(
            mask_path.stem
        ).lower()

        if key in mask_map:
            raise RuntimeError(
                "Multiple masks correspond to "
                f"the same image stem: {key}"
            )

        mask_map[key] = mask_path

    missing_masks = sorted(
        set(image_map) - set(mask_map)
    )

    extra_masks = sorted(
        set(mask_map) - set(image_map)
    )

    if missing_masks or extra_masks:

        messages = [
            f"Pairing problem detected in split: "
            f"{split_name}"
        ]

        if missing_masks:
            messages.append(
                "Images without masks: "
                + ", ".join(
                    missing_masks[:10]
                )
            )

        if extra_masks:
            messages.append(
                "Masks without images: "
                + ", ".join(
                    extra_masks[:10]
                )
            )

        raise RuntimeError(
            "\n".join(messages)
        )

    pairs = [
        ImageMaskPair(
            image_path=image_map[key],
            mask_path=mask_map[key],
            sample_id=image_map[key].stem,
        )
        for key in sorted(image_map)
    ]

    return pairs


# =============================================================================
# 6. IMAGE / MASK PREPROCESSING
# =============================================================================

def read_image(
    path: Path,
) -> np.ndarray:

    if path.suffix.lower() in {
        ".tif",
        ".tiff",
    }:
        return np.asarray(
            tifffile.imread(path)
        )

    try:
        from PIL import Image

    except ImportError as exc:
        raise ImportError(
            "Pillow is required for non-TIFF "
            "images. Install with: pip install pillow"
        ) from exc

    with Image.open(path) as image:
        return np.asarray(image)


def to_grayscale(
    array: np.ndarray,
    path: Path,
) -> np.ndarray:

    array = np.squeeze(array)

    if array.ndim == 2:
        return array

    if array.ndim == 3:

        if (
            array.shape[0] in (3, 4)
            and array.shape[-1]
            not in (3, 4)
        ):
            array = np.moveaxis(
                array,
                0,
                -1,
            )

        if array.shape[-1] in (3, 4):

            rgb = array[
                ..., :3
            ].astype(np.float32)

            return (
                0.2126 * rgb[..., 0]
                + 0.7152 * rgb[..., 1]
                + 0.0722 * rgb[..., 2]
            )

    raise ValueError(
        f"Unsupported image shape "
        f"{array.shape} for file: {path}"
    )


def normalize_sem_image(
    image: np.ndarray,
) -> np.ndarray:

    image = image.astype(
        np.float32,
        copy=False,
    )

    finite = np.isfinite(image)

    if not finite.any():
        raise ValueError(
            "Image contains no finite values."
        )

    if not finite.all():

        median_value = float(
            np.median(image[finite])
        )

        image = np.where(
            finite,
            image,
            median_value,
        )

    low = float(
        np.percentile(
            image,
            LOW_PERCENTILE,
        )
    )

    high = float(
        np.percentile(
            image,
            HIGH_PERCENTILE,
        )
    )

    if high <= low:

        raw_min = float(image.min())
        raw_max = float(image.max())

        if raw_max <= raw_min:
            return np.zeros_like(
                image,
                dtype=np.float32,
            )

        low = raw_min
        high = raw_max

    image = np.clip(
        image,
        low,
        high,
    )

    image = (
        image - low
    ) / (
        high - low
    )

    return image.astype(
        np.float32,
        copy=False,
    )


def prepare_binary_mask(
    mask: np.ndarray,
    path: Path,
) -> np.ndarray:

    mask = np.squeeze(mask)

    if mask.ndim == 3:

        if mask.shape[-1] in (3, 4):
            mask = mask[..., 0]

        elif mask.shape[0] in (3, 4):
            mask = mask[0]

    if mask.ndim != 2:
        raise ValueError(
            f"Mask must be 2D. "
            f"Got {mask.shape}: {path}"
        )

    if not np.isfinite(mask).all():
        raise ValueError(
            f"Mask contains NaN or Inf: {path}"
        )

    if np.any(mask < 0):
        raise ValueError(
            f"Mask contains negative labels: {path}"
        )

    unique_values = np.unique(mask)

    if unique_values.size > 16:
        raise ValueError(
            "Mask appears non-categorical with "
            f"{unique_values.size} values: {path}"
        )

    return (
        mask > 0
    ).astype(np.float32)


def load_pair(
    pair: ImageMaskPair,
) -> Tuple[np.ndarray, np.ndarray]:

    image = to_grayscale(
        read_image(pair.image_path),
        pair.image_path,
    )

    mask = prepare_binary_mask(
        read_image(pair.mask_path),
        pair.mask_path,
    )

    if image.shape != mask.shape:

        raise ValueError(
            f"Shape mismatch for "
            f"{pair.sample_id}: "
            f"image={image.shape}, "
            f"mask={mask.shape}"
        )

    image = normalize_sem_image(
        image
    )

    return image, mask


# =============================================================================
# 7. PATCH UTILITIES
# =============================================================================

def pad_if_needed(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:

    patch_h, patch_w = patch_size

    height, width = image.shape

    pad_bottom = max(
        0,
        patch_h - height,
    )

    pad_right = max(
        0,
        patch_w - width,
    )

    if (
        pad_bottom == 0
        and pad_right == 0
    ):
        return image, mask

    image_mode = (
        "reflect"
        if height > 1 and width > 1
        else "edge"
    )

    image = np.pad(
        image,
        (
            (0, pad_bottom),
            (0, pad_right),
        ),
        mode=image_mode,
    )

    mask = np.pad(
        mask,
        (
            (0, pad_bottom),
            (0, pad_right),
        ),
        mode="constant",
        constant_values=0,
    )

    return image, mask


def sliding_positions(
    full_length: int,
    patch_length: int,
    stride: int,
) -> List[int]:

    if full_length <= patch_length:
        return [0]

    positions = list(
        range(
            0,
            full_length
            - patch_length
            + 1,
            stride,
        )
    )

    final_position = (
        full_length
        - patch_length
    )

    if positions[-1] != final_position:
        positions.append(
            final_position
        )

    return positions


# =============================================================================
# 8. TRAINING AUGMENTATION
# =============================================================================

def augment_train_patch(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:

    if rng.random() < 0.5:

        image = np.flip(
            image,
            axis=1,
        )

        mask = np.flip(
            mask,
            axis=1,
        )

    if rng.random() < 0.5:

        image = np.flip(
            image,
            axis=0,
        )

        mask = np.flip(
            mask,
            axis=0,
        )

    rotation_k = int(
        rng.integers(
            0,
            4,
        )
    )

    if rotation_k:

        image = np.rot90(
            image,
            k=rotation_k,
        )

        mask = np.rot90(
            mask,
            k=rotation_k,
        )

    if rng.random() < 0.5:

        contrast = float(
            rng.uniform(
                0.90,
                1.10,
            )
        )

        brightness = float(
            rng.uniform(
                -0.05,
                0.05,
            )
        )

        image = (
            image * contrast
            + brightness
        )

    if rng.random() < 0.25:

        noise_sigma = float(
            rng.uniform(
                0.0,
                0.02,
            )
        )

        noise = rng.normal(
            0.0,
            noise_sigma,
            image.shape,
        ).astype(np.float32)

        image = image + noise

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return (
        np.ascontiguousarray(
            image,
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            mask,
            dtype=np.float32,
        ),
    )


# =============================================================================
# 9. COVERAGE CATEGORIES
# =============================================================================

def coverage_category(
    foreground_fraction: float,
) -> str:

    fraction = float(
        foreground_fraction
    )

    if fraction < 0.25:
        return "low_0_25"

    if fraction < 0.75:
        return "mixed_25_75"

    if fraction < 0.99:
        return "high_75_99"

    return "almost_all_99_100"


# =============================================================================
# 10. RUN-BALANCED + COVERAGE-AWARE TRAINING DATASET
# =============================================================================

class TrainingPatchDataset(Dataset):

    def __init__(
        self,
        pairs: Sequence[ImageMaskPair],
        patch_size: Tuple[int, int]
        = PATCH_SIZE,
        patches_per_image: int
        = TRAIN_PATCHES_PER_IMAGE,
        seed: int = SEED,
    ) -> None:

        self.pairs = list(pairs)

        self.patch_size = tuple(
            patch_size
        )

        self.patches_per_image = int(
            patches_per_image
        )

        self.seed = int(seed)

        self.epoch = 0

        self.cache: Dict[
            int,
            Tuple[
                np.ndarray,
                np.ndarray,
            ],
        ] = {}

        # Run -> list of training image indices.
        self.run_to_pair_indices = (
            defaultdict(list)
        )

        for pair_index, pair in enumerate(
            self.pairs
        ):

            run_name = extract_run(
                pair.sample_id
            )

            self.run_to_pair_indices[
                run_name
            ].append(
                pair_index
            )

        self.run_names = sorted(
            self.run_to_pair_indices.keys(),
            key=run_sort_key,
        )

        if len(self.run_names) == 0:
            raise RuntimeError(
                "No Runs were discovered "
                "in the training dataset."
            )

        # Run -> Category -> Pair -> Candidates
        self.candidate_pools = {
            run_name: {
                category: defaultdict(list)
                for category
                in CATEGORY_NAMES
            }
            for run_name
            in self.run_names
        }

        self._build_candidate_pools()
        self._validate_candidate_pools()

        total_samples = len(self)

        if (
            total_samples
            % len(self.run_names)
            != 0
        ):
            raise RuntimeError(
                "Training samples per epoch "
                "cannot be divided evenly "
                f"across {len(self.run_names)} Runs. "
                f"Total samples={total_samples}"
            )

        self.samples_per_run = (
            total_samples
            // len(self.run_names)
        )

    def __len__(self) -> int:

        return (
            len(self.pairs)
            * self.patches_per_image
        )

    def set_epoch(
        self,
        epoch: int,
    ) -> None:

        self.epoch = int(epoch)

    def _rng(
        self,
        index: int,
    ) -> np.random.Generator:

        sample_seed = (
            self.seed
            + self.epoch * 1_000_003
            + index * 10_007
        ) % (
            2**63 - 1
        )

        return np.random.default_rng(
            sample_seed
        )

    def _get_full_pair(
        self,
        pair_index: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
    ]:

        if pair_index not in self.cache:

            image, mask = load_pair(
                self.pairs[pair_index]
            )

            image, mask = (
                pad_if_needed(
                    image,
                    mask,
                    self.patch_size,
                )
            )

            self.cache[
                pair_index
            ] = (
                image,
                mask,
            )

        return self.cache[
            pair_index
        ]

    def _build_candidate_pools(
        self,
    ) -> None:

        patch_h, patch_w = (
            self.patch_size
        )

        stride_h, stride_w = (
            CANDIDATE_STRIDE
        )

        print("")
        print(
            "Building run-balanced "
            "coverage-aware candidate pools..."
        )

        for pair_index, pair in enumerate(
            self.pairs
        ):

            run_name = extract_run(
                pair.sample_id
            )

            _, mask = (
                self._get_full_pair(
                    pair_index
                )
            )

            height, width = mask.shape

            top_positions = (
                sliding_positions(
                    height,
                    patch_h,
                    stride_h,
                )
            )

            left_positions = (
                sliding_positions(
                    width,
                    patch_w,
                    stride_w,
                )
            )

            local_counts = {
                category: 0
                for category
                in CATEGORY_NAMES
            }

            for top in top_positions:

                for left in left_positions:

                    mask_patch = mask[
                        top
                        : top + patch_h,
                        left
                        : left + patch_w,
                    ]

                    foreground_fraction = (
                        float(
                            mask_patch.mean()
                        )
                    )

                    category = (
                        coverage_category(
                            foreground_fraction
                        )
                    )

                    candidate = (
                        PatchCandidate(
                            pair_index=(
                                pair_index
                            ),
                            top=int(top),
                            left=int(left),
                            foreground_fraction=(
                                foreground_fraction
                            ),
                        )
                    )

                    self.candidate_pools[
                        run_name
                    ][
                        category
                    ][
                        pair_index
                    ].append(
                        candidate
                    )

                    local_counts[
                        category
                    ] += 1

            print(
                f"  {pair.sample_id:<30} | "
                f"low="
                f"{local_counts['low_0_25']:4d} | "
                f"mixed="
                f"{local_counts['mixed_25_75']:4d} | "
                f"high="
                f"{local_counts['high_75_99']:4d} | "
                f"all="
                f"{local_counts['almost_all_99_100']:4d}"
            )

    def _validate_candidate_pools(
        self,
    ) -> None:

        print("")
        print(
            "Available coverage categories "
            "within each Run"
        )
        print("-" * 86)

        for run_name in self.run_names:

            available = []

            counts = {}

            for category in CATEGORY_NAMES:

                pair_dictionary = (
                    self.candidate_pools[
                        run_name
                    ][
                        category
                    ]
                )

                number_of_candidates = sum(
                    len(candidates)
                    for candidates
                    in pair_dictionary.values()
                )

                counts[
                    category
                ] = number_of_candidates

                if number_of_candidates > 0:
                    available.append(
                        category
                    )

            if not available:
                raise RuntimeError(
                    f"{run_name} has no "
                    "training patch candidates."
                )

            print(
                f"{run_name:<8} | "
                f"low={counts['low_0_25']:4d} | "
                f"mixed={counts['mixed_25_75']:4d} | "
                f"high={counts['high_75_99']:4d} | "
                f"all={counts['almost_all_99_100']:4d} | "
                f"available={len(available)} categories"
            )

        print("-" * 86)

    def _target_run(
        self,
        index: int,
    ) -> Tuple[str, int]:

        """
        Assign every epoch index deterministically
        to one Run.

        With 864 samples and 9 Runs:
            96 samples / Run exactly.
        """

        number_of_runs = len(
            self.run_names
        )

        run_position = (
            index % number_of_runs
        )

        within_run_position = (
            index // number_of_runs
        )

        run_name = self.run_names[
            run_position
        ]

        return (
            run_name,
            within_run_position,
        )

    def _available_categories(
        self,
        run_name: str,
    ) -> List[str]:

        available = []

        for category in CATEGORY_NAMES:

            pair_dictionary = (
                self.candidate_pools[
                    run_name
                ][
                    category
                ]
            )

            if any(
                len(candidates) > 0
                for candidates
                in pair_dictionary.values()
            ):
                available.append(
                    category
                )

        return available

    def _target_category(
        self,
        run_name: str,
        within_run_position: int,
    ) -> str:

        """
        Cycle uniformly among the categories that
        actually exist inside the selected Run.

        We do NOT force a Run to provide a
        morphology category that does not exist.
        """

        available = (
            self._available_categories(
                run_name
            )
        )

        if not available:
            raise RuntimeError(
                f"No available categories "
                f"for {run_name}"
            )

        category_index = (
            within_run_position
            % len(available)
        )

        return available[
            category_index
        ]

    def _sample_candidate(
        self,
        run_name: str,
        category: str,
        rng: np.random.Generator,
    ) -> PatchCandidate:

        """
        Within Run + Category:

        1. Select an eligible image uniformly.
        2. Select one patch from that image uniformly.

        This prevents one image from dominating only
        because it has many more candidate locations.
        """

        pair_dictionary = (
            self.candidate_pools[
                run_name
            ][
                category
            ]
        )

        eligible_pair_indices = [
            pair_index
            for pair_index, candidates
            in pair_dictionary.items()
            if len(candidates) > 0
        ]

        if not eligible_pair_indices:
            raise RuntimeError(
                f"No candidates for "
                f"{run_name} / {category}"
            )

        pair_choice = int(
            rng.integers(
                0,
                len(
                    eligible_pair_indices
                ),
            )
        )

        pair_index = (
            eligible_pair_indices[
                pair_choice
            ]
        )

        candidates = (
            pair_dictionary[
                pair_index
            ]
        )

        candidate_choice = int(
            rng.integers(
                0,
                len(candidates),
            )
        )

        return candidates[
            candidate_choice
        ]

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, object]:

        rng = self._rng(index)

        (
            run_name,
            within_run_position,
        ) = self._target_run(
            index
        )

        category = (
            self._target_category(
                run_name,
                within_run_position,
            )
        )

        candidate = (
            self._sample_candidate(
                run_name=run_name,
                category=category,
                rng=rng,
            )
        )

        pair_index = (
            candidate.pair_index
        )

        pair = self.pairs[
            pair_index
        ]

        image, mask = (
            self._get_full_pair(
                pair_index
            )
        )

        patch_h, patch_w = (
            self.patch_size
        )

        top = candidate.top
        left = candidate.left

        image_patch = image[
            top
            : top + patch_h,
            left
            : left + patch_w,
        ]

        mask_patch = mask[
            top
            : top + patch_h,
            left
            : left + patch_w,
        ]

        image_patch, mask_patch = (
            augment_train_patch(
                image_patch,
                mask_patch,
                rng,
            )
        )

        return {
            "image": torch.from_numpy(
                image_patch
            ).unsqueeze(0),

            "mask": torch.from_numpy(
                mask_patch
            ).unsqueeze(0),

            "sample_id": (
                pair.sample_id
            ),

            "run": run_name,

            "top": int(top),

            "left": int(left),

            "sampling_category": (
                category
            ),

            "foreground_fraction": (
                float(
                    mask_patch.mean()
                )
            ),
        }


# =============================================================================
# 11. VALIDATION / TEST SLIDING-WINDOW DATASET
# =============================================================================

class SlidingWindowDataset(Dataset):

    def __init__(
        self,
        pairs: Sequence[ImageMaskPair],
        patch_size: Tuple[int, int],
        stride: Tuple[int, int],
    ) -> None:

        self.pairs = list(pairs)

        self.patch_size = (
            patch_size
        )

        self.stride = stride

        self.cache: Dict[
            int,
            Tuple[
                np.ndarray,
                np.ndarray,
            ],
        ] = {}

        self.locations: List[
            PatchLocation
        ] = []

        self._build_index()

    def _build_index(
        self,
    ) -> None:

        patch_h, patch_w = (
            self.patch_size
        )

        stride_h, stride_w = (
            self.stride
        )

        for pair_index, pair in enumerate(
            self.pairs
        ):

            image = to_grayscale(
                read_image(
                    pair.image_path
                ),
                pair.image_path,
            )

            mask = prepare_binary_mask(
                read_image(
                    pair.mask_path
                ),
                pair.mask_path,
            )

            if image.shape != mask.shape:
                raise ValueError(
                    f"Shape mismatch for "
                    f"{pair.sample_id}: "
                    f"image={image.shape}, "
                    f"mask={mask.shape}"
                )

            original_height, original_width = (
                image.shape
            )

            padded_height = max(
                original_height,
                patch_h,
            )

            padded_width = max(
                original_width,
                patch_w,
            )

            top_positions = (
                sliding_positions(
                    padded_height,
                    patch_h,
                    stride_h,
                )
            )

            left_positions = (
                sliding_positions(
                    padded_width,
                    patch_w,
                    stride_w,
                )
            )

            for top in top_positions:

                for left in left_positions:

                    self.locations.append(
                        PatchLocation(
                            pair_index=(
                                pair_index
                            ),
                            top=top,
                            left=left,
                            original_height=(
                                original_height
                            ),
                            original_width=(
                                original_width
                            ),
                        )
                    )

    def __len__(
        self,
    ) -> int:

        return len(
            self.locations
        )

    def _get_full_pair(
        self,
        pair_index: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
    ]:

        if pair_index not in self.cache:

            image, mask = load_pair(
                self.pairs[
                    pair_index
                ]
            )

            image, mask = (
                pad_if_needed(
                    image,
                    mask,
                    self.patch_size,
                )
            )

            self.cache[
                pair_index
            ] = (
                image,
                mask,
            )

        return self.cache[
            pair_index
        ]

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, object]:

        location = (
            self.locations[
                index
            ]
        )

        pair = self.pairs[
            location.pair_index
        ]

        image, mask = (
            self._get_full_pair(
                location.pair_index
            )
        )

        patch_h, patch_w = (
            self.patch_size
        )

        image_patch = image[
            location.top
            : location.top + patch_h,
            location.left
            : location.left + patch_w,
        ]

        mask_patch = mask[
            location.top
            : location.top + patch_h,
            location.left
            : location.left + patch_w,
        ]

        image_patch = (
            np.ascontiguousarray(
                image_patch,
                dtype=np.float32,
            )
        )

        mask_patch = (
            np.ascontiguousarray(
                mask_patch,
                dtype=np.float32,
            )
        )

        return {
            "image": torch.from_numpy(
                image_patch
            ).unsqueeze(0),

            "mask": torch.from_numpy(
                mask_patch
            ).unsqueeze(0),

            "sample_id": (
                pair.sample_id
            ),

            "top": (
                location.top
            ),

            "left": (
                location.left
            ),

            "original_height": (
                location.original_height
            ),

            "original_width": (
                location.original_width
            ),
        }


# =============================================================================
# 12. DATALOADER FACTORY
# =============================================================================

def create_dataloaders():

    seed_everything(SEED)

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            "Dataset folder was not found.\n"
            f"Expected location:\n"
            f"{DATASET_DIR}"
        )

    train_pairs = discover_pairs(
        "train"
    )

    val_pairs = discover_pairs(
        "val"
    )

    test_pairs = discover_pairs(
        "test"
    )

    train_dataset = (
        TrainingPatchDataset(
            pairs=train_pairs,
            patch_size=PATCH_SIZE,
            patches_per_image=(
                TRAIN_PATCHES_PER_IMAGE
            ),
            seed=SEED,
        )
    )

    val_dataset = (
        SlidingWindowDataset(
            pairs=val_pairs,
            patch_size=PATCH_SIZE,
            stride=VAL_STRIDE,
        )
    )

    test_dataset = (
        SlidingWindowDataset(
            pairs=test_pairs,
            patch_size=PATCH_SIZE,
            stride=TEST_STRIDE,
        )
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        SEED
    )

    common_arguments = {
        "num_workers": (
            NUM_WORKERS
        ),
        "pin_memory": (
            torch.cuda.is_available()
        ),
        "worker_init_fn": (
            seed_worker
        ),
        "generator": (
            generator
        ),
        "persistent_workers": (
            NUM_WORKERS > 0
        ),
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=(
            TRAIN_BATCH_SIZE
        ),
        shuffle=True,
        drop_last=False,
        **common_arguments,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=(
            EVAL_BATCH_SIZE
        ),
        shuffle=False,
        drop_last=False,
        **common_arguments,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=(
            EVAL_BATCH_SIZE
        ),
        shuffle=False,
        drop_last=False,
        **common_arguments,
    )

    print_summary(
        train_pairs,
        val_pairs,
        test_pairs,
        train_dataset,
        val_dataset,
        test_dataset,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# =============================================================================
# 13. SUMMARY
# =============================================================================

def print_summary(
    train_pairs,
    val_pairs,
    test_pairs,
    train_dataset,
    val_dataset,
    test_dataset,
) -> None:

    print("")
    print("=" * 86)
    print(
        ""
        "RUN-BALANCED + COVERAGE-AWARE DATALOADER"
    )
    print("=" * 86)

    print(
        f"Train image-mask pairs:       "
        f"{len(train_pairs)}"
    )

    print(
        f"Validation pairs:             "
        f"{len(val_pairs)}"
    )

    print(
        f"Test pairs:                   "
        f"{len(test_pairs)}"
    )

    print(
        f"Experimental Runs in Train:   "
        f"{len(train_dataset.run_names)}"
    )

    print(
        f"Patch size:                   "
        f"{PATCH_SIZE}"
    )

    print(
        f"Training patches / epoch:     "
        f"{len(train_dataset)}"
    )

    print(
        f"Training patches / Run:       "
        f"{train_dataset.samples_per_run}"
    )

    print(
        f"Validation patches:           "
        f"{len(val_dataset)}"
    )

    print(
        f"Test patches:                 "
        f"{len(test_dataset)}"
    )

    print(
        f"Candidate stride:             "
        f"{CANDIDATE_STRIDE}"
    )

    print(
        "Training sampling:           "
        "Equal Run representation + "
        "within-Run coverage diversity"
    )

    print(
        "Training augmentation:       "
        "Enabled"
    )

    print(
        "Validation augmentation:     "
        "Disabled"
    )

    print(
        "Test augmentation:           "
        "Disabled"
    )

    print(
        f"Random seed:                  "
        f"{SEED}"
    )

    print(
        f"CUDA available:               "
        f"{torch.cuda.is_available()}"
    )

    print("=" * 86)


# =============================================================================
# 14. BATCH VALIDATION
# =============================================================================

def validate_batch(
    split_name: str,
    batch: Dict[str, object],
) -> None:

    images: Tensor = (
        batch["image"]
    )

    masks: Tensor = (
        batch["mask"]
    )

    if (
        images.ndim != 4
        or masks.ndim != 4
    ):
        raise RuntimeError(
            f"{split_name}: expected "
            f"BCHW tensors. "
            f"image={images.shape}, "
            f"mask={masks.shape}"
        )

    if images.shape != masks.shape:
        raise RuntimeError(
            f"{split_name}: shape mismatch "
            f"{images.shape} vs {masks.shape}"
        )

    if not torch.isfinite(
        images
    ).all():
        raise RuntimeError(
            f"{split_name}: images "
            "contain NaN/Inf."
        )

    if (
        images.min() < 0
        or images.max() > 1
    ):
        raise RuntimeError(
            f"{split_name}: normalized "
            "image outside [0,1]."
        )

    mask_values = torch.unique(
        masks
    )

    if not torch.all(
        (mask_values == 0)
        | (mask_values == 1)
    ):
        raise RuntimeError(
            f"{split_name}: mask is "
            f"not binary: "
            f"{mask_values.tolist()}"
        )

    print(
        f"{split_name:<10} | "
        f"images={tuple(images.shape)} | "
        f"masks={tuple(masks.shape)} | "
        f"image_range=("
        f"{images.min().item():.3f}, "
        f"{images.max().item():.3f}) | "
        f"mask_values="
        f"{mask_values.tolist()}"
    )


# =============================================================================
# 15. TRAINING DISTRIBUTION SANITY CHECK
# =============================================================================

def validate_training_distribution(
    train_loader,
) -> None:

    dataset = (
        train_loader.dataset
    )

    dataset.set_epoch(1)

    run_counts = defaultdict(int)

    run_category_counts = defaultdict(
        lambda: {
            category: 0
            for category
            in CATEGORY_NAMES
        }
    )

    category_counts = defaultdict(
        int
    )

    foreground_values = []

    for index in range(
        len(dataset)
    ):

        sample = dataset[
            index
        ]

        run_name = sample[
            "run"
        ]

        category = sample[
            "sampling_category"
        ]

        foreground_fraction = float(
            sample[
                "foreground_fraction"
            ]
        )

        run_counts[
            run_name
        ] += 1

        run_category_counts[
            run_name
        ][
            category
        ] += 1

        category_counts[
            category
        ] += 1

        foreground_values.append(
            foreground_fraction
        )

    values = np.asarray(
        foreground_values,
        dtype=float,
    )

    print("")
    print(
        "TRAINING "
        "SAMPLING SANITY CHECK"
    )
    print("=" * 86)

    print("")
    print(
        "RUN-LEVEL REPRESENTATION"
    )
    print("-" * 86)

    expected_per_run = (
        len(dataset)
        / len(
            dataset.run_names
        )
    )

    for run_name in (
        dataset.run_names
    ):

        count = run_counts[
            run_name
        ]

        percentage = (
            100.0
            * count
            / len(dataset)
        )

        categories = (
            run_category_counts[
                run_name
            ]
        )

        print(
            f"{run_name:<8} "
            f"{count:4d} patches "
            f"({percentage:6.2f}%) | "
            f"low="
            f"{categories['low_0_25']:3d} | "
            f"mixed="
            f"{categories['mixed_25_75']:3d} | "
            f"high="
            f"{categories['high_75_99']:3d} | "
            f"all="
            f"{categories['almost_all_99_100']:3d}"
        )

        if count != int(
            expected_per_run
        ):
            raise RuntimeError(
                f"{run_name} does not have "
                "the expected equal number "
                "of training patches."
            )

    print("-" * 86)

    print("")
    print(
        "GLOBAL COVERAGE-CATEGORY REPRESENTATION"
    )
    print("-" * 86)

    for category in CATEGORY_NAMES:

        count = category_counts[
            category
        ]

        percentage = (
            100.0
            * count
            / len(dataset)
        )

        print(
            f"{category:<24} "
            f"{count:4d} patches "
            f"({percentage:6.2f}%)"
        )

    print("")

    print(
        f"Mean sampled foreground:    "
        f"{values.mean() * 100:.2f}%"
    )

    print(
        f"Median sampled foreground:  "
        f"{np.median(values) * 100:.2f}%"
    )

    print(
        f"Minimum sampled foreground: "
        f"{values.min() * 100:.2f}%"
    )

    print(
        f"Maximum sampled foreground: "
        f"{values.max() * 100:.2f}%"
    )

    print("=" * 86)

    print(
        "Run-balance check: PASSED"
    )


# =============================================================================
# 16. MAIN / SMOKE TEST
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse an optional dataset path for the standalone smoke test."""
    parser = argparse.ArgumentParser(
        description=(
            "Build run-balanced and coverage-aware PyTorch DataLoaders "
            "for binary SEM segmentation."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help=(
            "Dataset root containing train/, val/, and test/ folders. "
            "Default: <script_dir>/dataset_split_10um"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    set_dataset_dir(args.dataset_dir)

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders()

    print("")
    print(
        "Running DataLoader "
        "smoke test..."
    )
    print("")

    validate_batch(
        "Train",
        next(
            iter(train_loader)
        ),
    )

    validate_batch(
        "Validation",
        next(
            iter(val_loader)
        ),
    )

    validate_batch(
        "Test",
        next(
            iter(test_loader)
        ),
    )

    validate_training_distribution(
        train_loader
    )

    print("")
    print(
        "DataLoader smoke test "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
