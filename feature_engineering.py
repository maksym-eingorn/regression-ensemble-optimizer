# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# feature_engineering.py

import random
import warnings
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

import config


def apply_autofeat(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_transform: np.ndarray,
    seed: int = config.RANDOM_SEED
) -> tuple[np.ndarray, np.ndarray, Any]:
    """
    Fit AutoFeat on X_fit and y_fit, then transform both X_fit and
    X_transform.
    """
    from autofeat import AutoFeatRegressor

    random.seed(seed)
    np.random.seed(seed)

    warnings.filterwarnings(
        "ignore", category=FutureWarning, module="autofeat"
    )

    afreg = AutoFeatRegressor(
        feateng_steps=config.AUTOFEAT_FEATENG_STEPS,
        featsel_runs=config.AUTOFEAT_FEATSEL_RUNS,
        n_jobs=config.AUTOFEAT_N_JOBS,
        verbose=0
    )

    X_fit_fe = afreg.fit_transform(X_fit, y_fit)
    X_transform_fe = afreg.transform(X_transform)

    return X_fit_fe.to_numpy(), X_transform_fe.to_numpy(), afreg


def scale_feature_engineered_data(
    X_dev_fe: np.ndarray, X_test_fe: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale AutoFeat-generated development and test feature matrices."""
    scaler_fe = StandardScaler()
    X_dev_fe_scaled = scaler_fe.fit_transform(X_dev_fe)
    X_test_fe_scaled = scaler_fe.transform(X_test_fe)

    return X_dev_fe_scaled, X_test_fe_scaled, scaler_fe
