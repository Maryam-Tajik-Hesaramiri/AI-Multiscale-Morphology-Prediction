# AI-Driven Multiscale Morphology Prediction

An AI-driven framework for connecting **manufacturing process parameters, SEM image morphology, and machine learning** to characterize and predict material structures.

This research focuses on hydrothermally grown ZnO structures and combines computer vision, deep learning, statistical analysis, and experimental design to develop a data-driven materials processing workflow.

## Project Overview

Understanding how processing conditions influence material morphology is an important challenge in advanced manufacturing and materials engineering.

This project develops a computational workflow to:

- Segment ZnO structures from SEM images using deep learning
- Quantify morphology from microscopy data
- Connect processing parameters to morphological features
- Train machine-learning models for morphology prediction
- Identify influential processing parameters
- Guide new experimental conditions for model validation

## Workflow

**Processing Parameters → SEM Imaging → U-Net Segmentation → Morphology Extraction → Machine Learning → Experimental Validation**

The experimental inputs include:

- Stirring speed (RPM)
- pH
- Growth time
- Growth temperature

SEM images are analyzed at multiple magnifications to capture morphology across different length scales.

## Computer Vision & Deep Learning

A U-Net-based semantic segmentation pipeline was developed to distinguish ZnO structures from the substrate in SEM images.

The segmentation workflow includes:

- Manual ground-truth annotation
- Train/validation/test dataset splitting
- Patch-based SEM image processing
- Data augmentation
- U-Net semantic segmentation
- Dice and IoU evaluation
- Morphological feature extraction

## Machine Learning

Extracted image features are combined with experimental processing parameters to construct a materials-processing dataset.

Machine-learning models are evaluated using **Leave-One-Run-Out cross-validation** to reduce data leakage between SEM images collected from the same experimental condition.

Current predictive modeling includes:

- Extra Trees regression
- Gaussian Process regression
- Model comparison and cross-validation
- Parameter sensitivity analysis
- Response-surface analysis
- Experimental validation design

## Technologies

**Programming & Data Analysis**

`Python` `NumPy` `Pandas` `scikit-learn`

**Deep Learning & Computer Vision**

`PyTorch` `U-Net` `Semantic Segmentation` `Image Processing`

**Materials & Manufacturing**

`SEM` `ZnO` `Hydrothermal Synthesis` `Design of Experiments` `Taguchi Method`

## Repository Structure

The repository will contain selected code and documentation for the major stages of the workflow:

```text
AI-Multiscale-Morphology-Prediction/
│
├── segmentation/
│   └── U-Net training and evaluation
│
├── morphology_analysis/
│   └── SEM feature extraction
│
├── machine_learning/
│   └── Predictive modeling and cross-validation
│
├── experimental_validation/
│   └── Candidate selection and validation design
│
├── figures/
│   └── Selected workflow and result visualizations
│
└── README.md
