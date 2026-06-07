# storage.py

from pathlib import Path

import joblib
import numpy as np


def save_numpy_arrays(
    output_dir: str,
    arrays: dict[str, np.ndarray]
) -> None:
    """Save multiple NumPy arrays into an output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, array in arrays.items():
        np.save(output_path / f"{name}.npy", array)


def load_numpy_arrays(
    output_dir: str,
    names: list[str]
) -> dict[str, np.ndarray]:
    """Load multiple NumPy arrays from an output directory."""
    output_path = Path(output_dir)

    return {
        name: np.load(output_path / f"{name}.npy")
        for name in names
    }


def save_objects(
    output_dir: str,
    objects: dict[str, object]
) -> None:
    """Save multiple Python objects with joblib."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, obj in objects.items():
        joblib.dump(obj, output_path / f"{name}.pkl", compress=3)


def load_objects(
    output_dir: str,
    names: list[str]
) -> dict[str, object]:
    """Load multiple Python objects saved with joblib."""
    output_path = Path(output_dir)

    return {
        name: joblib.load(output_path / f"{name}.pkl")
        for name in names
    }
