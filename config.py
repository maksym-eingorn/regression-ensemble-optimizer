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

# --- Output paths ---

PREPARED_DATA_DIR = "prepared_data"
