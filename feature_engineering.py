# feature_engineering.py

import random
import warnings
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

import config


def apply_autofeat(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, Any]:
    """
    Fit AutoFeat on the development set and transform development/test data.

    AutoFeat is fit only on the development set to avoid test data leakage.
    """
    from autofeat import AutoFeatRegressor

    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="autofeat"
    )

    afreg = AutoFeatRegressor(
        feateng_steps=config.AUTOFEAT_FEATENG_STEPS,
        featsel_runs=config.AUTOFEAT_FEATSEL_RUNS,
        n_jobs=config.AUTOFEAT_N_JOBS,
        verbose=0
    )

    X_dev_fe = afreg.fit_transform(X_dev, y_dev)
    X_test_fe = afreg.transform(X_test)

    return X_dev_fe.to_numpy(), X_test_fe.to_numpy(), afreg


def scale_feature_engineered_data(
    X_dev_fe: np.ndarray,
    X_test_fe: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale AutoFeat-generated development and test feature matrices."""
    scaler_fe = StandardScaler()
    X_dev_fe_scaled = scaler_fe.fit_transform(X_dev_fe)
    X_test_fe_scaled = scaler_fe.transform(X_test_fe)

    return X_dev_fe_scaled, X_test_fe_scaled, scaler_fe
