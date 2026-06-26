# config.py

# --- General settings ---

RANDOM_SEED = 42

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

# --- Output paths ---

PREPARED_DATA_DIR = "prepared_data"
OPTUNA_RESULTS_DIR = "optuna_results"
