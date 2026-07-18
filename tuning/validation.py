# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# tuning/validation.py

import numpy as np


def _to_numpy_array(array_like) -> np.ndarray:
    """Convert an array-like object to a NumPy array."""
    if isinstance(array_like, np.ndarray):
        return array_like

    if hasattr(array_like, "to_numpy"):
        return array_like.to_numpy()

    return np.asarray(array_like)


def validate_feature_target_arrays(
    X, y, X_name: str, y_name: str, sample_label: str = "n_samples"
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the feature matrix and the target vector."""
    X = _to_numpy_array(X)
    y = _to_numpy_array(y)

    if X.ndim != 2:
        raise ValueError(
            f"{X_name} must be 2D ({sample_label}, n_features), "
            f"got {X.ndim}D."
        )

    if y.ndim != 1:
        raise ValueError(
            f"{y_name} must be 1D ({sample_label},), got {y.ndim}D."
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Numbers of samples in {X_name} and {y_name} must match, "
            f"got {X.shape[0]} != {y.shape[0]}."
        )

    return X, y


def validate_regression_inputs(X_dev, y_dev) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return development features and target as NumPy arrays."""
    return validate_feature_target_arrays(
        X_dev,
        y_dev,
        "X_dev",
        "y_dev",
        sample_label="n_samples_dev"
    )


def validate_positive_integer(value: int, name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}.")


def validate_integer_range_settings(
    min_value: int, max_value: int, step: int, name: str
) -> None:
    """Validate integer search-range settings."""
    validate_positive_integer(min_value, f"{name}_min")
    validate_positive_integer(max_value, f"{name}_max")
    validate_positive_integer(step, f"{name}_step")

    if max_value < min_value:
        raise ValueError(
            f"{name}_max must be greater than or equal to {name}_min, "
            f"got {max_value} < {min_value}."
        )

    if (max_value - min_value) % step != 0:
        raise ValueError(
            f"{name}_step must evenly divide the range "
            f"{name}_max - {name}_min, got "
            f"({max_value} - {min_value}) % {step} != 0."
        )


def validate_kfold_settings(n_splits: int, n_samples: int) -> None:
    """Validate K-fold cross-validation settings."""
    validate_positive_integer(n_splits, "n_splits")

    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}.")

    if n_splits > n_samples:
        raise ValueError(
            f"n_splits cannot exceed the number of samples, "
            f"got n_splits={n_splits}, n_samples={n_samples}."
        )
