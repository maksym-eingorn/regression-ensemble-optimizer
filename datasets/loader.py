# datasets/loader.py

import numpy as np

from datasets.california_housing import load_california_housing


def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a regression dataset by name."""
    if dataset_name == "california_housing":
        return load_california_housing()

    raise ValueError(
        f"Unsupported dataset: {dataset_name}. "
        "Currently supported: california_housing."
    )
