# Regression Ensemble Optimizer

A modular Python machine learning project for building a regression ensemble optimization pipeline.

## Overview

The current version implements data preparation for supported regression datasets and XGBoost hyperparameter tuning with Optuna.

The data preparation pipeline includes dataset loading, development/test splitting, feature scaling, optional AutoFeat feature engineering, and saving prepared arrays and fitted preprocessing objects.

The XGBoost tuning workflow loads prepared development data, selects the configured feature set, runs Optuna with K-fold cross-validation, stores out-of-fold predictions for future ensemble search, and saves tuning results locally.

The project supports California Housing and Diabetes datasets. For California Housing, capped target values are removed before splitting.

Future updates will add additional model families, evaluation of selected models on the test set, ensemble search, and regression performance comparison.

## Project Structure

`main.py` — data preparation pipeline\
`run_xgboost_optuna.py` — XGBoost Optuna tuning workflow\
`config.py` — project settings and user-configurable parameters\
`environment.py` — numerical library thread settings for improved reproducibility\
`datasets/loader.py` — dataset dispatcher\
`datasets/california_housing.py` — California Housing dataset loading and cleaning\
`datasets/diabetes.py` — Diabetes dataset loading\
`preprocessing.py` — development/test splitting and standard feature scaling\
`feature_engineering.py` — optional AutoFeat feature engineering and scaling\
`storage.py` — saving and loading prepared NumPy arrays and fitted preprocessing objects\
`tuning/feature_sets.py` — feature set selection for tuning workflows\
`tuning/validation.py` — shared validation helpers for tuning inputs\
`tuning/result_storage.py` — saving and loading Optuna result artifacts\
`tuning/xgboost_optuna.py` — XGBoost Optuna tuning logic\
`requirements.txt` — Python package dependencies

## How It Works

The data preparation pipeline:

* loads the selected regression dataset
* removes capped target values for California Housing
* splits the data into development and test sets
* scales the original features
* optionally applies AutoFeat feature engineering
* scales the AutoFeat-generated features
* saves prepared arrays as `.npy` files
* saves fitted preprocessing objects as `.pkl` files

Generated data artifacts are saved locally in dataset-specific subfolders:

`prepared_data/california_housing/`\
`prepared_data/diabetes/`

The XGBoost Optuna tuning workflow:

* loads prepared data for the selected dataset
* chooses the configured feature set
* runs XGBoost hyperparameter tuning with Optuna
* uses K-fold cross-validation on the development set
* stores one out-of-fold prediction column per Optuna trial
* saves the completed Optuna study, trial numbers, out-of-fold predictions, RMSE values, and hyperparameters

Generated Optuna artifacts are saved locally in model-specific subfolders, such as:

`optuna_results/california_housing/original_scaled/xgboost/`

## How to Run

Install dependencies from the project directory with:

`pip install -r requirements.txt`

Then run:

`python main.py`

The script prepares the selected dataset and saves processed outputs into the corresponding dataset-specific subfolder within `prepared_data/`.

Next, run XGBoost Optuna tuning:

`python run_xgboost_optuna.py`

The tuning script loads the prepared development data, runs Optuna-based XGBoost tuning, and saves results into `optuna_results/`.

## Configuration

Main user-facing settings are stored in `config.py`.

Important data preparation settings include:

`DATASET_NAME` — selected dataset name\
`TEST_SIZE` — test set fraction\
`RANDOM_SEED` — random seed for reproducibility\
`USE_AUTOFEAT` — whether to apply AutoFeat feature engineering\
`PREPARED_DATA_DIR` — root output folder for prepared data

Important XGBoost Optuna settings are:

`XGBOOST_FEATURE_SET` — selected feature set for XGBoost tuning\
`XGBOOST_N_TRIALS` — number of Optuna trials\
`XGBOOST_N_JOBS` — number of parallel Optuna workers\
`XGBOOST_N_SPLITS` — number of K-fold cross-validation splits\
`XGBOOST_N_ESTIMATORS_MIN` — minimum number of XGBoost estimators considered by Optuna\
`XGBOOST_N_ESTIMATORS_MAX` — maximum number of XGBoost estimators considered by Optuna\
`XGBOOST_N_ESTIMATORS_STEP` — step size for the Optuna search over estimators\
`XGBOOST_VERBOSE` — whether to print trial-level RMSE values\
`OPTUNA_RESULTS_DIR` — root output folder for Optuna result artifacts

The currently supported datasets are:

`california_housing`\
`diabetes`

The currently supported feature sets are:

`original`\
`original_scaled`\
`autofeat`\
`autofeat_scaled`

## AutoFeat

If `USE_AUTOFEAT = True`, the pipeline fits AutoFeat on the development set and applies the learned transformation to both development and test sets.

AutoFeat is fit only on the development set to avoid test data leakage.

If `USE_AUTOFEAT = False`, the pipeline skips AutoFeat and saves only the original and scaled original feature matrices.

## XGBoost Optuna Tuning

The XGBoost tuning workflow uses Optuna to search over XGBoost hyperparameters, including the number of estimators.

Each Optuna trial trains one XGBoost configuration across all K folds and produces one full out-of-fold prediction vector for the development set.

The final out-of-fold prediction matrix has one column per trial and is saved for future ensemble search.

The test set is not used during Optuna tuning.

## Reproducibility

The project limits hidden parallelism in numerical libraries through `environment.py`.

For stricter reproducibility, `PYTHONHASHSEED` can be set before launching Python.

Optuna tuning uses a seeded sampler, seeded K-fold splitting, and seeded XGBoost models.

## Generated Files

The pipeline may generate files such as:

* `.npy` prepared arrays
* `.pkl` fitted preprocessing objects
* `.pkl` Optuna result artifacts
* the `prepared_data/` directory
* the `optuna_results/` directory

These files are ignored by Git because they are generated artifacts rather than source code.

## Planned Future Extensions

Future updates may add:

* evaluation of XGBoost on the test set
* LightGBM Optuna tuning
* ElasticNet tuning
* ensemble search
* regression metrics and model comparison

## Why This Project

This project is designed as a clean, extensible foundation for regression ensemble experimentation.

It emphasizes:

* modular Python architecture
* reproducible data preparation
* leakage-aware preprocessing
* configurable feature set selection
* Optuna-based hyperparameter tuning
* K-fold out-of-fold prediction generation
* clean separation of dataset loading, preprocessing, feature engineering, tuning, and storage
* a scalable structure for future model comparison and ensemble optimization
