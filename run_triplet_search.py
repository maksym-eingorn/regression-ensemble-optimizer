# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# run_triplet_search.py

from environment import configure_environment

configure_environment()

from pathlib import Path

import joblib
import numpy as np

import config
from ensemble.oof_matrix import build_oof_matrix, load_oof_result_block
from ensemble.triplet_search import find_top_triplet_ensembles
from storage import (
    get_data_split_name, get_dataset_output_dir, load_prepared_data
)
from tuning.feature_sets import feature_set_requires_autofeat


def _get_model_specs() -> list[tuple[str, str]]:
    """Return model names and configured feature sets for triplet search."""
    return [
        ("xgboost", config.XGBOOST_FEATURE_SET),
        ("lightgbm", config.LIGHTGBM_FEATURE_SET),
        ("elasticnet", config.ELASTICNET_FEATURE_SET)
    ]


def _get_triplet_output_dir(
    root_dir: str,
    dataset_name: str,
    data_split_seed: int,
    model_specs: list[tuple[str, str]]
) -> Path:
    """Return the output directory for triplet ensemble search results."""
    spec_name = "__".join(
        f"{model_name}-{feature_set}"
        for model_name, feature_set in model_specs
    )

    return (
        Path(root_dir)
        / dataset_name
        / get_data_split_name(data_split_seed)
        / spec_name
        / "triplet_ols"
    )


def main() -> None:
    """Run exact exhaustive OLS-weighted triplet ensemble search on saved
    out-of-fold predictions."""
    model_specs = _get_model_specs()

    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    include_autofeat = any(
        feature_set_requires_autofeat(feature_set)
        for _, feature_set in model_specs
    )

    arrays, _ = load_prepared_data(
        prepared_data_dir, include_autofeat=include_autofeat
    )

    y_dev = np.asarray(arrays["y_dev"], dtype=np.float64)

    blocks = [
        load_oof_result_block(
            config.OPTUNA_RESULTS_DIR,
            config.DATASET_NAME,
            config.DATA_SPLIT_SEED,
            feature_set,
            model_name
        )
        for model_name, feature_set in model_specs
    ]

    P, column_metadata = build_oof_matrix(blocks)

    triplet_metadata, triplet_oof_predictions = find_top_triplet_ensembles(
        P,
        y_dev,
        top_n=config.TRIPLET_TOP_N,
        n_threads=config.TRIPLET_N_THREADS,
        weight_l1_limit=config.TRIPLET_WEIGHT_L1_LIMIT
    )

    if not triplet_metadata:
        raise RuntimeError("Triplet search did not return any valid triplets.")

    for triplet in triplet_metadata:
        i = int(triplet["i"])
        j = int(triplet["j"])
        k = int(triplet["k"])

        triplet["models"] = [
            column_metadata[i],
            column_metadata[j],
            column_metadata[k]
        ]

    output_dir = _get_triplet_output_dir(
        config.ENSEMBLE_RESULTS_DIR,
        config.DATASET_NAME,
        config.DATA_SPLIT_SEED,
        model_specs
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        column_metadata,
        output_dir / "base_model_column_metadata.pkl",
        compress=3
    )
    joblib.dump(
        triplet_metadata,
        output_dir / "triplet_oof_metadata.pkl",
        compress=3
    )
    joblib.dump(
        triplet_oof_predictions.astype(np.float32),
        output_dir / "triplet_oof_predictions.pkl",
        compress=3
    )

    best_triplet = triplet_metadata[0]

    print(f"Unified out-of-fold prediction matrix shape: {P.shape}")
    print(
        "Triplet out-of-fold prediction matrix shape: "
        f"{triplet_oof_predictions.shape}"
    )

    print(f"Triplet search results saved to: {output_dir}")
    print()
    print(
        f"Best triplet OOF RMSE = {best_triplet['oof_rmse']:.5f}, "
        f"columns "
        f"({best_triplet['i']}, {best_triplet['j']}, {best_triplet['k']})"
    )
    print(
        "Best triplet weights: "
        f"({best_triplet['wi']:.5f}, "
        f"{best_triplet['wj']:.5f}, "
        f"{best_triplet['wk']:.5f})"
    )


if __name__ == "__main__":
    main()
