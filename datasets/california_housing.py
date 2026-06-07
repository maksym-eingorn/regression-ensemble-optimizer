# datasets/california_housing.py

import numpy as np
from sklearn.datasets import fetch_california_housing

import config


def load_california_housing() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the California Housing dataset and remove capped target values.

    The original target is the median house value in units of $100,000.
    The capped value 5.00001 corresponds to $500,001.
    """
    data = fetch_california_housing(as_frame=True)

    X = data.data
    y = data.target

    mask = y < config.CALIFORNIA_TARGET_CAP
    X = X[mask]
    y = y[mask]

    return X.to_numpy(), y.to_numpy()
