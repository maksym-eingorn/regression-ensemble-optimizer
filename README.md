# Regression Ensemble Optimizer

A modular Python machine learning project for building a regression ensemble optimization pipeline.

## Overview

The current version implements data preparation for supported regression datasets, XGBoost, LightGBM, and ElasticNet hyperparameter tuning with Optuna, and final XGBoost, LightGBM, and ElasticNet evaluation on the test set.

The data preparation pipeline includes dataset loading, development/test splitting, feature scaling, optional AutoFeat feature engineering, and saving prepared arrays and fitted preprocessing objects.

The XGBoost, LightGBM, and ElasticNet workflows load prepared development data, select the configured feature set, run Optuna with K-fold cross-validation, store out-of-fold predictions for future ensemble search, save tuning results locally, and evaluate the best tuned models on the held-out test set.

The project supports California Housing and Diabetes datasets. For California Housing, capped target values are removed before splitting.

Future updates will add ensemble search and regression performance comparison.

## Project Structure

`main.py` — data preparation pipeline\
`run_xgboost_optuna.py` — XGBoost Optuna tuning workflow\
`run_lightgbm_optuna.py` — LightGBM Optuna tuning workflow\
`run_elasticnet_optuna.py` — ElasticNet Optuna tuning workflow\
`evaluate_xgboost.py` — XGBoost evaluation on the test set\
`evaluate_lightgbm.py` — LightGBM evaluation on the test set\
`evaluate_elasticnet.py` — ElasticNet evaluation on the test set\
`config.py` — project settings and user-configurable parameters\
`environment.py` — numerical library thread settings for improved reproducibility\
`datasets/loader.py` — dataset dispatcher\
`datasets/california_housing.py` — California Housing dataset loading and cleaning\
`datasets/diabetes.py` — Diabetes dataset loading\
`preprocessing.py` — development/test splitting and standard feature scaling\
`feature_engineering.py` — optional AutoFeat feature engineering and scaling\
`storage.py` — saving and loading prepared NumPy arrays and fitted preprocessing objects\
`evaluation/validation.py` — validation helpers for final model evaluation\
`evaluation/xgboost_evaluation.py` — XGBoost retraining on the full development set and RMSE evaluation on the test set\
`evaluation/lightgbm_evaluation.py` — LightGBM retraining on the full development set and RMSE evaluation on the test set\
`evaluation/elasticnet_evaluation.py` — ElasticNet retraining on the full development set and RMSE evaluation on the test set\
`tuning/feature_sets.py` — feature set selection for tuning and evaluation workflows\
`tuning/validation.py` — shared validation helpers for feature and target arrays\
`tuning/result_storage.py` — saving and loading Optuna result artifacts\
`tuning/xgboost_optuna.py` — XGBoost Optuna tuning logic\
`tuning/lightgbm_optuna.py` — LightGBM Optuna tuning logic\
`tuning/elasticnet_optuna.py` — ElasticNet Optuna tuning logic\
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

The XGBoost, LightGBM, and ElasticNet Optuna tuning workflows:

* load prepared data for the selected dataset
* choose the configured feature set
* run model-specific hyperparameter tuning with Optuna
* use K-fold cross-validation on the development set
* store one out-of-fold prediction column per Optuna trial
* save the completed Optuna study, trial numbers, out-of-fold predictions, RMSE values, and hyperparameters

Generated Optuna artifacts are saved locally in model-specific subfolders, such as:

`optuna_results/california_housing/original_scaled/xgboost/`\
`optuna_results/california_housing/original_scaled/lightgbm/`\
`optuna_results/california_housing/autofeat_scaled/elasticnet/`

The XGBoost, LightGBM, and ElasticNet test evaluation workflows:

* load the prepared development and test data for the selected dataset
* choose the same configured feature set used during tuning
* load the saved model-specific Optuna study
* retrain the best model on the full development set
* evaluate the retrained model on the held-out test set
* print the final test RMSE

## How to Run

Install dependencies from the project directory with:

`pip install -r requirements.txt`

Then run:

`python main.py`

The script prepares the selected dataset and saves processed outputs into the corresponding dataset-specific subfolder within `prepared_data/`.

Next, run XGBoost Optuna tuning:

`python run_xgboost_optuna.py`

The XGBoost tuning script loads the prepared development data, runs Optuna-based XGBoost tuning, and saves results into `optuna_results/`.

Also run LightGBM Optuna tuning:

`python run_lightgbm_optuna.py`

The LightGBM tuning script loads the prepared development data, runs Optuna-based LightGBM tuning, and saves results into `optuna_results/`.

In addition, run ElasticNet Optuna tuning:

`python run_elasticnet_optuna.py`

The ElasticNet tuning script loads the prepared development data, runs Optuna-based ElasticNet tuning, and saves results into `optuna_results/`.

To evaluate the best tuned XGBoost model on the test set, run:

`python evaluate_xgboost.py`

To evaluate the best tuned LightGBM model on the test set, run:

`python evaluate_lightgbm.py`

To evaluate the best tuned ElasticNet model on the test set, run:

`python evaluate_elasticnet.py`

Each evaluation script loads the saved model-specific Optuna study, retrains the best model on the full development set, evaluates it on the held-out test set, and prints the test RMSE.

## Configuration

Main user-facing settings are stored in `config.py`.

Important data preparation settings include:

`DATASET_NAME` — selected dataset name\
`TEST_SIZE` — test set fraction\
`RANDOM_SEED` — random seed for reproducibility\
`USE_AUTOFEAT` — whether to apply AutoFeat feature engineering\
`PREPARED_DATA_DIR` — root output folder for prepared data

Important model-specific Optuna settings are grouped by model family.

For XGBoost, the settings use the `XGBOOST_` prefix. For LightGBM, the settings use the `LIGHTGBM_` prefix. For ElasticNet, the settings use the `ELASTICNET_` prefix.

`XGBOOST_FEATURE_SET` / `LIGHTGBM_FEATURE_SET` / `ELASTICNET_FEATURE_SET` — selected feature set for tuning and evaluation\
`XGBOOST_N_TRIALS` / `LIGHTGBM_N_TRIALS` / `ELASTICNET_N_TRIALS` — number of Optuna trials\
`XGBOOST_N_JOBS` / `LIGHTGBM_N_JOBS` / `ELASTICNET_N_JOBS` — number of parallel Optuna workers\
`XGBOOST_N_SPLITS` / `LIGHTGBM_N_SPLITS` / `ELASTICNET_N_SPLITS` — number of K-fold cross-validation splits\
`XGBOOST_VERBOSE` / `LIGHTGBM_VERBOSE` / `ELASTICNET_VERBOSE` — whether to print trial-level RMSE values

Important tree-model estimator settings are:

`XGBOOST_N_ESTIMATORS_MIN` / `LIGHTGBM_N_ESTIMATORS_MIN` — minimum number of estimators considered by Optuna\
`XGBOOST_N_ESTIMATORS_MAX` / `LIGHTGBM_N_ESTIMATORS_MAX` — maximum number of estimators considered by Optuna\
`XGBOOST_N_ESTIMATORS_STEP` / `LIGHTGBM_N_ESTIMATORS_STEP` — step size for the Optuna search over estimators

Important ElasticNet optimization settings are:

`ELASTICNET_MAX_ITER` — maximum number of ElasticNet optimization iterations\
`ELASTICNET_TOL` — optimization tolerance for ElasticNet

`OPTUNA_RESULTS_DIR` represents the root output folder for Optuna result artifacts.

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

The default ElasticNet configuration uses the `autofeat_scaled` feature set, so AutoFeat-generated outputs must be available before running ElasticNet Optuna tuning and test evaluation.

If `USE_AUTOFEAT = False`, the pipeline skips AutoFeat and saves only the original and scaled original feature matrices.

## XGBoost, LightGBM, and ElasticNet Optuna Tuning

The XGBoost, LightGBM, and ElasticNet tuning workflows use Optuna to search over model-specific hyperparameters.

Each Optuna trial trains one model configuration across all K folds and produces one full out-of-fold prediction vector for the development set.

The final out-of-fold prediction matrix has one column per trial and is saved for future ensemble search.

The test set is not used during Optuna tuning. After tuning is complete, the best saved configuration for each model family can be retrained on the full development set and evaluated once on the held-out test set.

## XGBoost, LightGBM, and ElasticNet Test Evaluation

The test evaluation workflows load the saved Optuna study for the configured dataset, feature set, and model family.

The best hyperparameters from the corresponding Optuna study are used to retrain a final model on the full development set.

The retrained model is then evaluated on the held-out test set, and the final test RMSE is printed.

This keeps the test set separate from hyperparameter tuning and uses it only for final model evaluation.

## Reproducibility

The project limits hidden parallelism in numerical libraries through `environment.py`.

For stricter reproducibility, `PYTHONHASHSEED` can be set before launching Python.

Optuna tuning uses a seeded sampler, seeded K-fold splitting, and seeded XGBoost, LightGBM, and ElasticNet models.

## Generated Files

The pipeline may generate files such as:

* `.npy` prepared arrays
* `.pkl` fitted preprocessing objects
* `.pkl` Optuna studies and result artifacts
* the `prepared_data/` directory
* the `optuna_results/` directory

These files are ignored by Git because they are generated artifacts rather than source code.

## Planned Future Extensions

Future updates may add:

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
* final evaluation on the held-out test set
* clean separation of dataset loading, preprocessing, feature engineering, tuning, evaluation, and storage
* a scalable structure for future model comparison and ensemble optimization
