# Regression Ensemble Optimizer

A modular Python machine learning project for building a regression ensemble optimization pipeline.

## Overview

The current version implements data preparation for supported regression datasets, XGBoost, LightGBM, and ElasticNet hyperparameter tuning with Optuna, final individual model evaluation on the held-out test set, exact OLS-weighted triplet ensemble search, triplet ensemble evaluation on the held-out test set, Caruana-style greedy ensemble selection, and greedy ensemble evaluation on the held-out test set.

The data preparation pipeline includes dataset loading, development/test splitting, feature scaling, optional AutoFeat feature engineering, and saving prepared arrays and fitted preprocessing objects.

The XGBoost, LightGBM, and ElasticNet workflows load prepared development data, select the configured feature set, run Optuna with K-fold cross-validation, store out-of-fold predictions for ensemble search, save tuning results locally, and evaluate the best tuned individual models on the held-out test set.

The triplet ensemble workflow combines saved out-of-fold prediction columns from configured model families, runs exact exhaustive triplet search with unconstrained OLS weights, optionally applies an L1 weight guard to discard high-cancellation triplets, saves the retained OOF-ranked triplets, and evaluates the saved triplet ensembles on the held-out test set.

The greedy ensemble workflow uses the same saved out-of-fold prediction matrix, builds a convex ensemble through Caruana-style greedy selection, allows repeated model selections when configured, converts selection counts into final ensemble weights, saves the selected greedy ensemble, and evaluates it on the held-out test set.

The project supports California Housing and Diabetes datasets. For California Housing, capped target values are removed before splitting.

## Project Structure

`main.py` — data preparation pipeline\
`run_xgboost_optuna.py` — XGBoost Optuna tuning workflow\
`run_lightgbm_optuna.py` — LightGBM Optuna tuning workflow\
`run_elasticnet_optuna.py` — ElasticNet Optuna tuning workflow\
`run_triplet_search.py` — exact OLS-weighted triplet ensemble search workflow\
`run_greedy_ensemble_selection.py` — Caruana-style greedy ensemble selection workflow\
`evaluate_xgboost.py` — XGBoost evaluation on the test set\
`evaluate_lightgbm.py` — LightGBM evaluation on the test set\
`evaluate_elasticnet.py` — ElasticNet evaluation on the test set\
`evaluate_triplet_ensembles.py` — saved triplet ensemble evaluation on the test set\
`evaluate_greedy_ensemble.py` — saved greedy ensemble evaluation on the test set\
`config.py` — project settings and user-configurable parameters\
`environment.py` — numerical library thread settings for improved reproducibility\
`datasets/loader.py` — dataset dispatcher\
`datasets/california_housing.py` — California Housing dataset loading and cleaning\
`datasets/diabetes.py` — Diabetes dataset loading\
`preprocessing.py` — development/test splitting and standard feature scaling\
`feature_engineering.py` — optional AutoFeat feature engineering and scaling\
`storage.py` — saving and loading prepared NumPy arrays and fitted preprocessing objects\
`ensemble/oof_matrix.py` — construction of a unified out-of-fold prediction matrix\
`ensemble/triplet_search.py` — Python wrapper for the native triplet search extension\
`ensemble/greedy_selection.py` — Caruana-style greedy ensemble selection logic\
`evaluation/validation.py` — validation helpers for final model evaluation\
`evaluation/xgboost_evaluation.py` — XGBoost retraining on the full development set and RMSE evaluation on the test set\
`evaluation/lightgbm_evaluation.py` — LightGBM retraining on the full development set and RMSE evaluation on the test set\
`evaluation/elasticnet_evaluation.py` — ElasticNet retraining on the full development set and RMSE evaluation on the test set\
`evaluation/ensemble_evaluation.py` — triplet and greedy ensemble retraining on the full development set and RMSE evaluation on the test set\
`tuning/feature_sets.py` — feature set selection for tuning and evaluation workflows\
`tuning/validation.py` — shared validation helpers for feature and target arrays\
`tuning/result_storage.py` — saving and loading Optuna result artifacts\
`tuning/xgboost_optuna.py` — XGBoost Optuna tuning logic\
`tuning/lightgbm_optuna.py` — LightGBM Optuna tuning logic\
`tuning/elasticnet_optuna.py` — ElasticNet Optuna tuning logic\
`native/triplet_search.cpp` — C++/OpenMP implementation of exact exhaustive triplet search\
`setup_triplet_search.py` — build script for the native triplet search extension\
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
* score each trial by full-vector OOF RMSE
* store one out-of-fold prediction column per Optuna trial
* save the completed Optuna study, trial numbers, out-of-fold predictions, OOF RMSE values, per-fold RMSE values, and hyperparameters

Generated Optuna artifacts are saved locally in model-specific subfolders, such as:

`optuna_results/california_housing/original_scaled/xgboost/`\
`optuna_results/california_housing/original_scaled/lightgbm/`\
`optuna_results/california_housing/autofeat_scaled/elasticnet/`

The individual model test evaluation workflows:

* load the prepared development and test data for the selected dataset
* choose the same configured feature set used during tuning
* load the saved model-specific Optuna study
* retrain the best model on the full development set
* evaluate the retrained model on the held-out test set
* print the final test RMSE

The triplet ensemble search workflow:

* loads saved out-of-fold predictions from the configured model families
* builds one unified out-of-fold prediction matrix
* searches all 3-model combinations exactly
* fits unconstrained OLS weights for each triplet on development set out-of-fold predictions
* ranks triplets by OOF RMSE
* optionally applies the configured L1 weight guard to discard high-cancellation triplets
* saves the retained OOF-ranked triplet metadata and out-of-fold predictions

The greedy ensemble selection workflow:

* loads saved out-of-fold predictions from the configured model families
* builds one unified out-of-fold prediction matrix
* runs Caruana-style greedy ensemble selection
* optionally restricts the candidate pool with `GREEDY_POOL_SIZE`
* allows repeated model selections when `GREEDY_ALLOW_REPEATS = True`
* converts selection counts into convex weights
* saves the greedy ensemble metadata and out-of-fold predictions

Generated ensemble search artifacts are saved locally within `ensemble_results/`.

The triplet ensemble test evaluation workflow:

* loads the saved OOF-ranked triplet metadata
* retrains the selected base models on the full development set
* combines their test predictions using the saved OLS weights
* evaluates the saved triplet ensembles on the held-out test set
* saves triplet test metadata and triplet test predictions
* prints the best OOF-ranked triplet and a diagnostic best-by-test summary among the retained triplets

The greedy ensemble test evaluation workflow:

* loads the saved greedy ensemble metadata
* retrains the selected base models on the full development set
* combines their test predictions using the saved convex weights
* evaluates the greedy ensemble on the held-out test set
* saves the greedy ensemble test metadata and test predictions
* prints OOF RMSE and test RMSE

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

Each individual model evaluation script loads the saved model-specific Optuna study, retrains the best model on the full development set, evaluates it on the held-out test set, and prints the test RMSE.

Before running triplet ensemble search, build the native C++ extension:

`python setup_triplet_search.py build_ext --inplace`

The native extension requires Eigen. If Eigen is not located in the default path expected by the build script, set the `EIGEN_INCLUDE_DIR` environment variable to the Eigen folder before building.

Then run exact OLS-weighted triplet ensemble search:

`python run_triplet_search.py`

The triplet search script loads saved Optuna out-of-fold predictions, builds a unified out-of-fold matrix, runs exact exhaustive triplet search, optionally applies the configured L1 weight guard, and saves the retained OOF-ranked triplet results into `ensemble_results/`.

Evaluate the saved triplet ensembles on the held-out test set:

`python evaluate_triplet_ensembles.py`

The triplet evaluation script retrains the selected base models on the full development set, combines their test predictions using the saved OLS weights, evaluates the triplets on the held-out test set, and saves the resulting test artifacts into `ensemble_results/`.

Also, run Caruana-style greedy ensemble selection:

`python run_greedy_ensemble_selection.py`

The greedy ensemble selection workflow does not require building the native C++ extension; it only requires the saved Optuna out-of-fold prediction artifacts.

Evaluate the saved greedy ensemble on the held-out test set:

`python evaluate_greedy_ensemble.py`

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

Important triplet ensemble search settings are:

`TRIPLET_TOP_N` — number of top unguarded OOF-ranked triplets requested from the native exhaustive search before optional Python-side filtering\
`TRIPLET_WEIGHT_L1_LIMIT` — optional maximum allowed L1 norm of the three OLS triplet weights; set to `None` to disable the guard\
`TRIPLET_N_THREADS` — number of OpenMP threads used by the native triplet search

Important greedy ensemble selection settings are:

`GREEDY_POOL_SIZE` — number of the best individual models considered by greedy selection; set to `None` to use the full model zoo\
`GREEDY_ENSEMBLE_SIZE` — number of greedy selection steps\
`GREEDY_ALLOW_REPEATS` — whether the same model may be selected multiple times

`OPTUNA_RESULTS_DIR` represents the root output folder for Optuna result artifacts.

`ENSEMBLE_RESULTS_DIR` represents the root output folder for ensemble search and ensemble evaluation artifacts.

The native triplet search build requires Eigen. By default, the build script looks for Eigen at `C:\Libraries\eigen-3.4.1`. If Eigen is installed elsewhere, set the `EIGEN_INCLUDE_DIR` environment variable to the Eigen folder before building.

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

Each Optuna trial trains one model configuration across all K folds and produces one full out-of-fold prediction vector for the development set. Each trial is scored by full-vector OOF RMSE; per-fold RMSE values are saved separately for diagnostics.

The final out-of-fold prediction matrix for each model family has one column per trial and is saved for triplet search and greedy ensemble selection workflows.

The test set is not used during Optuna tuning. After tuning is complete, the best saved configuration for each model family can be retrained on the full development set and evaluated once on the held-out test set.

## XGBoost, LightGBM, and ElasticNet Test Evaluation

The individual model test evaluation workflows load the saved Optuna study for the configured dataset, feature set, and model family.

The best hyperparameters from the corresponding Optuna study are used to retrain a final model on the full development set.

The retrained model is then evaluated on the held-out test set, and the final test RMSE is printed.

This keeps the test set separate from hyperparameter tuning and uses it only for final individual model evaluation.

## Triplet Ensemble Search

The triplet ensemble search workflow uses saved out-of-fold predictions from the configured XGBoost, LightGBM, and ElasticNet Optuna runs.

It builds a unified out-of-fold prediction matrix where each column corresponds to one saved Optuna trial. It then performs exact exhaustive search over all 3-column combinations. For each triplet, unconstrained OLS weights are fitted on the development set out-of-fold predictions, and the triplet is ranked by full-vector OOF RMSE. The native search returns the top unguarded OOF-ranked candidates; if `TRIPLET_WEIGHT_L1_LIMIT` is not `None`, Python-side filtering then discards triplets whose weight L1 norm exceeds the configured limit.

The test set is not used during the triplet search.

Triplet search artifacts are saved within `ensemble_results/` and include:

* `base_model_column_metadata.pkl`
* `triplet_oof_metadata.pkl`
* `triplet_oof_predictions.pkl`

## Greedy Ensemble Selection

The greedy ensemble selection workflow uses saved out-of-fold predictions from the configured XGBoost, LightGBM, and ElasticNet Optuna runs.

It builds the same unified out-of-fold prediction matrix used by triplet search, where each column corresponds to one saved Optuna trial. It then runs Caruana-style greedy ensemble selection. The algorithm first restricts the candidate set to the configured pool of the best individual models, unless the full model zoo is used. It then builds a convex ensemble by repeatedly adding the model that gives the lowest OOF MSE after averaging the selected prediction vectors. If repeats are enabled, the same model may be selected multiple times. Final convex weights are selection counts divided by the total number of completed greedy selections.

The test set is not used during the greedy ensemble selection.

Greedy selection artifacts are saved within `ensemble_results/` and include:

* `base_model_column_metadata.pkl`
* `greedy_oof_metadata.pkl`
* `greedy_oof_predictions.pkl`

## Triplet Ensemble Test Evaluation

The triplet ensemble test evaluation workflow loads the saved triplets, retrains the selected base models on the full development set, combines their test predictions using the saved OLS weights, and evaluates each triplet ensemble on the held-out test set.

The script reports the best OOF-ranked triplet and also prints the lowest test RMSE among the retained triplets as diagnostic information only. The test set should not be used to choose the final model selection rule.

Triplet test artifacts are saved within `ensemble_results/` and include:

* `triplet_test_metadata.pkl`
* `triplet_test_predictions.pkl`

## Greedy Ensemble Test Evaluation

The greedy ensemble test evaluation workflow loads the saved greedy ensemble metadata, retrains the selected base models on the full development set, combines their test predictions using the saved convex weights, and evaluates the greedy ensemble on the held-out test set.

The script reports the greedy ensemble OOF RMSE and test RMSE.

Greedy ensemble test artifacts are saved within `ensemble_results/` and include:

* `greedy_test_metadata.pkl`
* `greedy_test_predictions.pkl`

## Reproducibility

The project limits hidden parallelism in numerical libraries through `environment.py`.

For stricter reproducibility, `PYTHONHASHSEED` can be set before launching Python.

Optuna tuning uses a seeded sampler, seeded K-fold splitting, and seeded XGBoost, LightGBM, and ElasticNet models.

The native triplet search uses OpenMP for parallel exhaustive search. Its thread count is controlled by `TRIPLET_N_THREADS`.

## Generated Files

The pipeline may generate files such as:

* `.npy` prepared arrays
* `.pkl` fitted preprocessing objects
* `.pkl` Optuna studies and result artifacts
* `.pkl` ensemble search and ensemble evaluation artifacts
* compiled native extension files
* the `prepared_data/` directory
* the `optuna_results/` directory
* the `ensemble_results/` directory
* the `build/` directory

These files are ignored by Git because they are generated artifacts rather than source code.

## Why This Project

This project is designed as a clean, extensible foundation for regression ensemble experimentation.

It emphasizes:

* modular Python architecture
* reproducible data preparation
* leakage-aware preprocessing
* configurable feature set selection
* Optuna-based hyperparameter tuning
* K-fold out-of-fold prediction generation
* exact exhaustive ensemble search over saved out-of-fold predictions
* native C++/OpenMP acceleration for combinatorial triplet search
* Caruana-style greedy ensemble selection over saved out-of-fold predictions
* final evaluation on the held-out test set
* clean separation of dataset loading, preprocessing, feature engineering, tuning, ensemble search, evaluation, and storage
* a scalable structure for future model comparison and ensemble optimization
