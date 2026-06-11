# datasets/diabetes.py

import numpy as np
from sklearn.datasets import load_diabetes


def load_diabetes_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Load the Diabetes regression dataset."""
    data = load_diabetes(as_frame=True)

    X = data.data
    y = data.target

    return X.to_numpy(), y.to_numpy()
