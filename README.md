# AI-Driven Multiscale Morphology Prediction

A computational materials framework connecting **manufacturing process parameters, SEM image morphology, computer vision, and machine learning** to characterize and predict material structures.

This research focuses on hydrothermally grown ZnO and integrates experimental design, microscopy, deep-learning segmentation, quantitative morphology analysis, and small-data predictive modeling.

## Project Overview

Understanding how manufacturing conditions influence material morphology is an important challenge in materials processing and advanced manufacturing.

This project develops an end-to-end workflow to:

- Segment ZnO structures from SEM images using U-Net
- Quantify morphology from microscopy data
- Connect hydrothermal growth parameters to morphological features
- Compare machine-learning models for morphology prediction
- Evaluate models while preserving experimental independence
- Support the design of new experiments for external validation

## Workflow

**Growth Parameters → SEM Imaging → U-Net Segmentation → Morphology Quantification → Predictive Modeling → Experimental Validation**

The investigated processing parameters include:

- Stirring speed (RPM)
- pH
- Growth time
- Growth temperature

SEM images are collected across experimental runs and analyzed to quantify morphology at different length scales.

## U-Net Segmentation

A U-Net semantic-segmentation pipeline is used to distinguish ZnO structures from the substrate in SEM images.

The segmentation workflow includes:

- Manual ground-truth annotation
- Train/validation/test dataset splitting
- Patch-based SEM processing
- Run-balanced and coverage-aware training sampling
- U-Net training and evaluation
- Dice and IoU performance assessment
- Sliding-window full-image inference
- Reconstruction of full-resolution probability maps and binary masks
- Image-level ZnO coverage extraction

Overlapping patch probabilities are averaged before thresholding during full-image reconstruction.

## Morphology Prediction

Image-derived morphology measurements are combined with the experimental growth parameters:

**RPM + pH + Time + Temperature → ZnO Coverage + Mean Nanosheet Width**

The dataset contains repeated SEM observations from **9 independent experimental runs**. Images collected under the same growth condition are therefore not treated as independent experimental samples.

Model performance is evaluated using **Leave-One-Run-Out (LORO) cross-validation** to prevent information leakage between images from the same experimental condition.

Candidate models include:

- Linear Regression
- Ridge Regression
- Bayesian Ridge
- Random Forest
- Extra Trees
- Gradient Boosting
- Additive Gaussian Process
- Weighted ensemble

Performance is compared using MAE, RMSE, and R².

Hierarchical bootstrap analysis is also used to characterize prediction uncertainty while preserving the nested experimental structure.

## Technologies

**Programming & Machine Learning**

`Python` `NumPy` `Pandas` `scikit-learn` `PyTorch`

**Computer Vision**

`U-Net` `Semantic Segmentation` `Image Processing` `Sliding-Window Inference`

**Materials & Manufacturing**

`SEM` `ZnO` `Hydrothermal Synthesis` `Design of Experiments` `Taguchi Method`

## Repository Structure

```text
AI-Multiscale-Morphology-Prediction/
│
├── segmentation/
│   ├── dataset_split.py
│   ├── dataset_statistics.py
│   ├── prepare_dataloader.py
│   ├── audit_training_distribution.py
│   ├── unet.py
│   ├── losses_metrics.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict_full_dataset.py
│   └── README.md
│
├── prediction/
│   ├── build_ai_dataset.py
│   ├── train_morphology_predictor.py
│   └── README.md
│
├── .gitignore
└── README.md
```

### `segmentation/`

Contains the U-Net workflow for dataset preparation, training, evaluation, and full-resolution SEM inference.

### `prediction/`

Contains the workflow for combining image-derived morphology with experimental parameters and evaluating growth-parameter-to-morphology regression models.

## Current Scope

The repository contains selected research code illustrating the computational workflow rather than the complete raw experimental dataset.

Large SEM datasets, manually annotated masks, trained model checkpoints, and intermediate research outputs are excluded from version control.

The current predictive models are based on 9 independent DOE conditions and are intended primarily for interpretation and prediction within or near the investigated experimental domain.

External validation using newly synthesized conditions is planned as the next stage of the study.

## Research Goal

The long-term objective is to develop a data-driven framework that links:

**Manufacturing Conditions → Material Morphology → Predictive Models**

to support more systematic design and optimization of functional material-processing conditions.
