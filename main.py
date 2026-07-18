# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# main.py

from environment import configure_environment

configure_environment()

import numpy as np

import config
from datasets.loader import load_dataset
from feature_engineering import apply_autofeat, scale_feature_engineered_data
from preprocessing import split_dev_test, scale_dev_test
from storage import get_dataset_output_dir, save_numpy_arrays, save_objects


def _print_shapes(arrays: dict[str, np.ndarray]) -> None:
    """Print shapes of arrays for a quick sanity check."""
    for name, array in arrays.items():
        print(f"{name}: {array.shape}")


def main() -> None:
    """Prepare the selected regression dataset and save processed outputs."""
    X, y = load_dataset(config.DATASET_NAME)

    X_dev, X_test, y_dev, y_test = split_dev_test(X, y)

    X_dev_scaled, X_test_scaled, scaler = scale_dev_test(X_dev, X_test)

    arrays_to_save = {
        "X": X,
        "y": y,
        "X_dev": X_dev,
        "y_dev": y_dev,
        "X_test": X_test,
        "y_test": y_test,
        "X_dev_scaled": X_dev_scaled,
        "X_test_scaled": X_test_scaled
    }

    objects_to_save = {
        "scaler": scaler
    }

    if config.USE_AUTOFEAT:
        X_dev_fe, X_test_fe, afreg = apply_autofeat(X_dev, y_dev, X_test)

        X_dev_fe_scaled, X_test_fe_scaled, scaler_fe = (
            scale_feature_engineered_data(X_dev_fe, X_test_fe)
        )

        arrays_to_save.update({
            "X_dev_fe": X_dev_fe,
            "X_test_fe": X_test_fe,
            "X_dev_fe_scaled": X_dev_fe_scaled,
            "X_test_fe_scaled": X_test_fe_scaled
        })

        objects_to_save.update({
            "afreg": afreg,
            "scaler_fe": scaler_fe
        })

    output_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    save_numpy_arrays(output_dir, arrays_to_save)
    save_objects(output_dir, objects_to_save)

    print(f"Prepared data saved to: {output_dir}")
    print()
    _print_shapes(arrays_to_save)


if __name__ == "__main__":
    main()
