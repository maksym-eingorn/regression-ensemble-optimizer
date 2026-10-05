# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# config.py

import os

# --- General settings ---

RANDOM_SEED = 42

# Controls only the development/test split.
DATA_SPLIT_SEED = 100

# --- Dataset settings ---

# Supported datasets: "california_housing", "diabetes"
DATASET_NAME = "california_housing"
TEST_SIZE = 0.2

# The California Housing target is measured in units of $100,000.
# The capped value 5.00001 corresponds to $500,001.
CALIFORNIA_TARGET_CAP = 5.00001

# --- Feature engineering ---

# If True, create AutoFeat feature-engineered versions of the data.
# If False, only original features and scaled original features are saved.
USE_AUTOFEAT = True

AUTOFEAT_FEATENG_STEPS = 2
AUTOFEAT_FEATSEL_RUNS = 5
AUTOFEAT_N_JOBS = 1

# --- XGBoost Optuna tuning ---

# Supported feature sets:
# "original", "original_scaled", "autofeat", "autofeat_scaled"
XGBOOST_FEATURE_SET = "original_scaled"

XGBOOST_N_TRIALS = 1000
XGBOOST_N_JOBS = 1
XGBOOST_N_SPLITS = 5
XGBOOST_N_ESTIMATORS_MIN = 100
XGBOOST_N_ESTIMATORS_MAX = 1000
XGBOOST_N_ESTIMATORS_STEP = 50
XGBOOST_VERBOSE = True

# --- LightGBM Optuna tuning ---

# Supported feature sets:
# "original", "original_scaled", "autofeat", "autofeat_scaled"
LIGHTGBM_FEATURE_SET = "original_scaled"

LIGHTGBM_N_TRIALS = 1000
LIGHTGBM_N_JOBS = 1
LIGHTGBM_N_SPLITS = 5
LIGHTGBM_N_ESTIMATORS_MIN = 100
LIGHTGBM_N_ESTIMATORS_MAX = 1000
LIGHTGBM_N_ESTIMATORS_STEP = 50
LIGHTGBM_VERBOSE = True

# --- ElasticNet Optuna tuning ---

# Supported feature sets:
# "original", "original_scaled", "autofeat", "autofeat_scaled"
ELASTICNET_FEATURE_SET = "autofeat_scaled"

ELASTICNET_N_TRIALS = 100
ELASTICNET_N_JOBS = 1
ELASTICNET_N_SPLITS = 5
ELASTICNET_MAX_ITER = 20000
ELASTICNET_TOL = 1e-4
ELASTICNET_VERBOSE = True

# --- Triplet ensemble search ---

TRIPLET_TOP_N = 1000

# Fixed shrinkage of unrestricted OLS triplet weights toward equal weights.
# 0.0 uses pure OLS weights.
# 1.0 uses pure equal weights (1/3, 1/3, 1/3).
TRIPLET_ALPHA = 0.0

# Maximum allowed L1 norm of the final alpha-blended triplet weights.
# Set to None to disable the guard.
TRIPLET_WEIGHT_L1_LIMIT = 2.0

# Use all available logical CPU cores except one for the native C++/OpenMP
# triplet search. The C++ RAII guard applies this only during the search call
# and then restores the previous OpenMP thread setting.
TRIPLET_N_THREADS = max(1, (os.cpu_count() or 2) - 1)

# --- Greedy ensemble selection ---

# Number of the best individual models considered by greedy selection.
# Set to None to use the full model zoo.
GREEDY_POOL_SIZE = None

# Number of greedy selection steps. With repeats enabled, this is the total
# number of model selections, not necessarily the number of unique models.
GREEDY_ENSEMBLE_SIZE = 3

# Caruana-style greedy ensemble selection allows the same model to be selected
# multiple times; final convex weights are selection counts divided by the
# total number of selections.
GREEDY_ALLOW_REPEATS = True

# --- Output paths ---

PREPARED_DATA_DIR = "prepared_data"
OPTUNA_RESULTS_DIR = "optuna_results"
ENSEMBLE_RESULTS_DIR = "ensemble_results"
