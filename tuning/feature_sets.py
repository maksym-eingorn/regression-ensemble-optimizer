# tuning/feature_sets.py

import numpy as np


SUPPORTED_FEATURE_SETS = {
    "original": "X_dev",
    "original_scaled": "X_dev_scaled",
    "autofeat": "X_dev_fe",
    "autofeat_scaled": "X_dev_fe_scaled",
}


def validate_feature_set(feature_set: str) -> None:
    """Validate the selected feature set name."""
    if feature_set not in SUPPORTED_FEATURE_SETS:
        supported = ", ".join(SUPPORTED_FEATURE_SETS)
        raise ValueError(
            f"Unsupported feature set: {feature_set}. "
            f"Currently supported: {supported}."
        )


def feature_set_requires_autofeat(feature_set: str) -> bool:
    """Return whether the selected feature set requires AutoFeat outputs."""
    validate_feature_set(feature_set)

    return feature_set in {"autofeat", "autofeat_scaled"}


def get_development_data(
    arrays: dict[str, np.ndarray], feature_set: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select development features and target from prepared arrays.

    Parameters
    ----------
    arrays
        Dictionary of prepared NumPy arrays.
    feature_set
        Feature set name.

    Returns
    -------
    X_dev : np.ndarray
        Selected development feature matrix.
    y_dev : np.ndarray
        Development target vector.
    """
    validate_feature_set(feature_set)

    X_key = SUPPORTED_FEATURE_SETS[feature_set]
    y_key = "y_dev"

    if X_key not in arrays:
        raise KeyError(
            f"Required feature array '{X_key}' was not found in prepared data."
        )

    if y_key not in arrays:
        raise KeyError(
            f"Required target array '{y_key}' was not found in prepared data."
        )

    return arrays[X_key], arrays[y_key]
