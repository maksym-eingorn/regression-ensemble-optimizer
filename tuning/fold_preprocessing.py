# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# tuning/fold_preprocessing.py

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from feature_engineering import apply_autofeat
from tuning.feature_sets import validate_feature_set


@dataclass(frozen=True)
class PreparedFold:
    """Fold-specific feature matrices and their original row indices."""

    train_indices: np.ndarray
    valid_indices: np.ndarray
    X_train: np.ndarray
    X_valid: np.ndarray


def prepare_kfold_feature_data(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    feature_set: str,
    n_splits: int,
    seed: int
) -> tuple[PreparedFold, ...]:
    """
    Prepare fold-specific feature matrices for out-of-fold model training.

    Preprocessing is fitted only on each fold's training rows and then applied
    to that fold's validation rows. The prepared folds can be reused across
    all Optuna trials because preprocessing does not depend on model
    hyperparameters.
    """
    validate_feature_set(feature_set)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    prepared_folds = []

    for train_indices, valid_indices in kf.split(X_dev):
        X_train_raw = X_dev[train_indices]
        X_valid_raw = X_dev[valid_indices]
        y_train = y_dev[train_indices]

        if feature_set == "original":
            X_train = X_train_raw
            X_valid = X_valid_raw

        elif feature_set == "original_scaled":
            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train_raw)
            X_valid = scaler.transform(X_valid_raw)

        elif feature_set == "autofeat":
            X_train, X_valid, _ = apply_autofeat(
                X_train_raw, y_train, X_valid_raw, seed=seed
            )

        else:
            X_train_fe, X_valid_fe, _ = apply_autofeat(
                X_train_raw, y_train, X_valid_raw, seed=seed
            )

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train_fe)
            X_valid = scaler.transform(X_valid_fe)

        prepared_folds.append(
            PreparedFold(
                train_indices=train_indices.copy(),
                valid_indices=valid_indices.copy(),
                X_train=np.asarray(X_train, dtype=np.float64),
                X_valid=np.asarray(X_valid, dtype=np.float64)
            )
        )

    return tuple(prepared_folds)
