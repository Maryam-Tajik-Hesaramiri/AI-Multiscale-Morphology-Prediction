# Morphology Prediction

This module develops data-driven models to predict ZnO morphology from hydrothermal growth parameters.

## Inputs

The models use four synthesis parameters:

* RPM
* pH
* Growth time
* Temperature

## Prediction Targets

Two morphology descriptors are modeled:

* ZnO surface coverage (%), obtained from U-Net segmentation
* Mean nanosheet width (µm), obtained from SEM image analysis

## Experimental Structure

The dataset contains 458 SEM images collected from 9 independent DOE runs.

Because multiple SEM images belong to the same experimental condition, the images are treated as repeated measurements rather than independent samples.

Model performance is therefore evaluated using **Leave-One-Run-Out (LORO) cross-validation**, where one complete experimental run is held out during each validation fold.

## Workflow

`build_ai_dataset.py`

Combines:

* DOE growth parameters
* U-Net predicted coverage
* SEM-derived nanosheet width measurements

and generates image-level and run-level morphology datasets.

`train_morphology_predictor.py`

Compares several small-data regression approaches:

* Linear Regression
* Ridge Regression
* Bayesian Ridge
* Random Forest
* Extra Trees
* Gradient Boosting
* Additive Gaussian Process
* Weighted ensemble

Models are evaluated using MAE, RMSE, and R² across the LORO folds.

A hierarchical bootstrap is also used to characterize prediction uncertainty while preserving the nested experimental structure.

## Outputs

The pipeline produces:

* LORO prediction tables
* Model comparison metrics
* Selected predictors for coverage and width
* Bootstrap confidence intervals
* Training-domain information
* Model-comparison figures

## Modeling Note

The current dataset contains only 9 independent synthesis conditions. Therefore, the models are intended primarily for interpretation and prediction within or near the studied DOE domain.

Prospective experiments at new synthesis conditions are required for external validation.
