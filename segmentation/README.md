# U-Net Segmentation Pipeline

This directory contains the deep-learning pipeline used to segment ZnO morphology from SEM images.

The segmentation stage converts microscopy images into binary morphology maps that can subsequently be used for quantitative feature extraction and process–morphology modeling.

## Workflow

The segmentation pipeline follows the sequence:

**Dataset Split → Dataset QC → Patch Sampling → U-Net Training → Independent Test Evaluation**

The workflow was designed to account for substantial variation in ZnO surface coverage and morphology across experimental runs.

## Pipeline Components

### `dataset_split.py`
Creates run-aware training, validation, and independent test subsets from manually labeled SEM images.

### `dataset_statistics.py`
Analyzes dataset composition and mask coverage distributions to identify imbalance across images and experimental runs.

### `prepare_dataloader.py`
Builds the patch-based PyTorch data pipeline.

Training uses:

- 256 × 256 image patches
- run-balanced sampling
- coverage-aware patch sampling
- equal sampling contribution from each experimental run

Validation and test images are evaluated using overlapping patches.

### `audit_training_distribution.py`
Audits the effective training distribution produced by the sampler and verifies representation across experimental runs and morphology-coverage ranges.

### `unet.py`
Defines the U-Net architecture used for binary semantic segmentation of ZnO morphology in SEM images.

The network uses:

- grayscale SEM input
- encoder–decoder architecture
- skip connections
- Group Normalization
- dropout regularization
- single-channel segmentation output

### `losses_metrics.py`
Implements the loss functions and segmentation metrics used during training and evaluation.

The training objective combines:

- Binary Cross-Entropy (BCE)
- Dice loss

Evaluation metrics include:

- Dice coefficient
- Intersection over Union (IoU)
- Precision
- Recall

### `train.py`
Trains the U-Net using the run-balanced, coverage-aware patch sampling strategy.

The training pipeline includes:

- AdamW optimization
- validation monitoring
- early stopping
- best-checkpoint selection
- training-history export

### `evaluate.py`
Evaluates the trained model on the independent test set.

Full-image predictions are reconstructed from overlapping patches using Hann-weighted probability blending before thresholding.

The evaluation pipeline produces:

- predicted segmentation masks
- probability maps
- error maps
- per-image segmentation metrics
- overall test metrics
- probability diagnostics

## Why Run-Balanced Sampling?

The SEM dataset contains substantial differences in morphology and ZnO coverage between experimental runs.

Uniform random patch sampling can cause runs containing more candidate patches or highly covered regions to dominate training.

Run-balanced sampling gives each experimental condition equal representation during an epoch, while coverage-aware sampling exposes the model to a broader range of sparse, intermediate, and dense morphology.

## Evaluation Strategy

The independent test set is kept separate from model training and validation.

Predictions are generated patch-by-patch and reconstructed into full-resolution probability maps. Overlapping predictions are combined using Hann-weighted blending to reduce patch-boundary artifacts.

Final binary masks are then compared with manual ground-truth annotations using Dice, IoU, precision, and recall.

## Role in the Multiscale Framework

This segmentation pipeline represents the computer-vision stage of the broader project:

**Processing Parameters → SEM Imaging → Morphology Segmentation → Feature Extraction → Machine Learning → Experimental Validation**

The resulting segmentation maps provide the basis for quantitative morphology analysis and subsequent modeling of relationships between hydrothermal processing conditions and ZnO morphology.

---

This repository contains selected research code intended to demonstrate the computational workflow. Raw microscopy data and research datasets are not publicly distributed.
