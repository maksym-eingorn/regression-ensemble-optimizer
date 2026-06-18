# tuning/validation.py

import numpy as np


def validate_regression_inputs(X_dev, y_dev) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return development features and target as NumPy arrays."""
    if not isinstance(X_dev, np.ndarray):
        if hasattr(X_dev, "to_numpy"):
            X_dev = X_dev.to_numpy()
        else:
            X_dev = np.asarray(X_dev)

    if not isinstance(y_dev, np.ndarray):
        if hasattr(y_dev, "to_numpy"):
            y_dev = y_dev.to_numpy()
        else:
            y_dev = np.asarray(y_dev)

    if X_dev.ndim != 2:
        raise ValueError(
            f"X_dev must be 2D (n_samples_dev, n_features), got {X_dev.ndim}D."
        )

    if y_dev.ndim != 1:
        raise ValueError(
            f"y_dev must be 1D (n_samples_dev,), got {y_dev.ndim}D."
        )

    if X_dev.shape[0] != y_dev.shape[0]:
        raise ValueError(
            f"Numbers of samples in X_dev and y_dev must match, "
            f"got {X_dev.shape[0]} != {y_dev.shape[0]}."
        )

    return X_dev, y_dev


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
