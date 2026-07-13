# storage.py

from pathlib import Path

import joblib
import numpy as np


BASE_ARRAY_NAMES = [
    "X",
    "y",
    "X_dev",
    "y_dev",
    "X_test",
    "y_test",
    "X_dev_scaled",
    "X_test_scaled"
]

AUTOFEAT_ARRAY_NAMES = [
    "X_dev_fe",
    "X_test_fe",
    "X_dev_fe_scaled",
    "X_test_fe_scaled"
]

BASE_OBJECT_NAMES = [
    "scaler"
]

AUTOFEAT_OBJECT_NAMES = [
    "afreg",
    "scaler_fe"
]


def validate_data_split_seed(data_split_seed: int) -> None:
    """Validate the development/test split seed."""
    if (not isinstance(data_split_seed, int) or
            isinstance(data_split_seed, bool)):
        raise ValueError(
            f"data_split_seed must be an integer, got {data_split_seed}."
        )

    if data_split_seed < 0:
        raise ValueError(
            f"data_split_seed must be non-negative, got {data_split_seed}."
        )


def get_data_split_name(data_split_seed: int) -> str:
    """Return the folder name for one development/test split."""
    validate_data_split_seed(data_split_seed)
    return f"split_seed_{data_split_seed}"


def get_dataset_output_dir(
    root_dir: str | Path, dataset_name: str, data_split_seed: int
) -> Path:
    """Return the output directory for a specific prepared dataset split."""
    return Path(root_dir) / dataset_name / get_data_split_name(data_split_seed)


def get_prepared_array_names(include_autofeat: bool) -> list[str]:
    """Return the expected prepared NumPy array names."""
    names = BASE_ARRAY_NAMES.copy()

    if include_autofeat:
        names.extend(AUTOFEAT_ARRAY_NAMES)

    return names


def get_prepared_object_names(include_autofeat: bool) -> list[str]:
    """Return the expected fitted preprocessing object names."""
    names = BASE_OBJECT_NAMES.copy()

    if include_autofeat:
        names.extend(AUTOFEAT_OBJECT_NAMES)

    return names


def save_numpy_arrays(
    output_dir: str | Path, arrays: dict[str, np.ndarray]
) -> None:
    """Save multiple NumPy arrays into an output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, array in arrays.items():
        np.save(output_path / f"{name}.npy", array)


def load_numpy_arrays(
    output_dir: str | Path, names: list[str]
) -> dict[str, np.ndarray]:
    """Load multiple NumPy arrays from an output directory."""
    output_path = Path(output_dir)

    return {name: np.load(output_path / f"{name}.npy") for name in names}


def save_objects(
    output_dir: str | Path, objects: dict[str, object]
) -> None:
    """Save multiple Python objects with joblib."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, obj in objects.items():
        joblib.dump(obj, output_path / f"{name}.pkl", compress=3)


def load_objects(
    output_dir: str | Path, names: list[str]
) -> dict[str, object]:
    """Load multiple Python objects saved with joblib."""
    output_path = Path(output_dir)

    return {name: joblib.load(output_path / f"{name}.pkl") for name in names}


def load_prepared_data(
    output_dir: str | Path, include_autofeat: bool
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load prepared arrays and fitted preprocessing objects."""
    array_names = get_prepared_array_names(include_autofeat)
    object_names = get_prepared_object_names(include_autofeat)

    arrays = load_numpy_arrays(output_dir, array_names)
    objects = load_objects(output_dir, object_names)

    return arrays, objects
