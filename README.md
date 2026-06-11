# Regression Ensemble Optimizer

A modular Python machine learning project for building a regression ensemble optimization pipeline.

## Overview

The current version implements the initial data preparation stage for supported regression datasets, including dataset loading, development/test splitting, feature scaling, optional AutoFeat feature engineering, and saving prepared arrays and fitted preprocessing objects.

The project supports California Housing and Diabetes datasets. For California Housing, capped target values are removed before splitting.

Future updates will add Optuna-based model tuning, out-of-fold prediction generation, ensemble search, and regression performance comparison.

## Project Structure

`main.py` — program entry point and data preparation workflow\
`config.py` — project settings and user-configurable parameters\
`environment.py` — numerical library thread settings for improved reproducibility\
`datasets/loader.py` — dataset dispatcher\
`datasets/california_housing.py` — California Housing dataset loading and cleaning\
`datasets/diabetes.py` — Diabetes dataset loading\
`preprocessing.py` — development/test splitting and standard feature scaling\
`feature_engineering.py` — optional AutoFeat feature engineering and scaling\
`storage.py` — saving and loading NumPy arrays and fitted preprocessing objects\
`requirements.txt` — Python package dependencies

## How It Works

The current pipeline:

- loads the selected regression dataset
- removes capped target values for California Housing
- splits the data into development and test sets
- scales the original features
- optionally applies AutoFeat feature engineering
- scales the AutoFeat-generated features
- saves prepared arrays as `.npy` files
- saves fitted preprocessing objects as `.pkl` files

Generated data artifacts are saved locally in:

`prepared_data/`

## How to Run

Install dependencies from the project directory with:

`pip install -r requirements.txt`

Then run:

`python main.py`

The script prepares the selected dataset and saves processed outputs into `prepared_data/`.

## Configuration

Main user-facing settings are stored in `config.py`.

Important settings include:

`DATASET_NAME` — selected dataset name\
`TEST_SIZE` — test set fraction\
`RANDOM_SEED` — random seed for reproducibility\
`USE_AUTOFEAT` — whether to apply AutoFeat feature engineering\
`PREPARED_DATA_DIR` — output folder for prepared data

The currently supported datasets are:

`california_housing`\
`diabetes`

## AutoFeat

If `USE_AUTOFEAT = True`, the pipeline fits AutoFeat on the development set and applies the learned transformation to both development and test sets.

AutoFeat is fit only on the development set to avoid test data leakage.

If `USE_AUTOFEAT = False`, the pipeline skips AutoFeat and saves only the original and scaled original feature matrices.

## Reproducibility

The project limits hidden parallelism in numerical libraries through `environment.py`.

For stricter reproducibility, `PYTHONHASHSEED` can be set before launching Python.

## Generated Files

The pipeline may generate files such as:

- `.npy` arrays
- `.pkl` fitted preprocessing objects
- the `prepared_data/` directory

These files are ignored by Git because they are generated artifacts rather than source code.

## Planned Future Extensions

Future updates may add:

- Optuna hyperparameter optimization
- out-of-fold prediction generation
- ensemble search
- regression metrics and model comparison

## Why This Project

This project is designed as a clean, extensible foundation for regression ensemble experimentation.

It emphasizes:

- modular Python architecture
- reproducible data preparation
- leakage-aware preprocessing
- optional automated feature engineering
- clean separation of dataset loading, preprocessing, feature engineering, and storage
- a scalable structure for future Optuna tuning and ensemble optimization
