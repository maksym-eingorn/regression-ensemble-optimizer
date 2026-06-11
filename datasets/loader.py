# datasets/loader.py

import numpy as np

from datasets.california_housing import load_california_housing
from datasets.diabetes import load_diabetes_dataset


def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a regression dataset by name."""
    if dataset_name == "california_housing":
        return load_california_housing()

    if dataset_name == "diabetes":
        return load_diabetes_dataset()

    raise ValueError(
        f"Unsupported dataset: {dataset_name}. "
        "Currently supported: california_housing, diabetes."
    )
