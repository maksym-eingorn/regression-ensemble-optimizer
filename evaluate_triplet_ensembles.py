# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# evaluate_triplet_ensembles.py

from environment import configure_environment

configure_environment()

from typing import cast

import joblib

import config
from evaluation.ensemble_evaluation import evaluate_triplet_ensembles_on_test
from storage import get_dataset_output_dir, load_prepared_data
from ensemble.triplet_results import get_triplet_output_dir
from tuning.feature_sets import feature_set_requires_autofeat


def _get_model_specs() -> list[tuple[str, str]]:
    """Return model names and configured feature sets for triplet evaluation."""
    return [
        ("xgboost", config.XGBOOST_FEATURE_SET),
        ("lightgbm", config.LIGHTGBM_FEATURE_SET),
        ("elasticnet", config.ELASTICNET_FEATURE_SET)
    ]


def main() -> None:
    """Evaluate saved top triplet ensembles on the held-out test set."""
    model_specs = _get_model_specs()

    triplet_output_dir = get_triplet_output_dir(
        config.ENSEMBLE_RESULTS_DIR,
        config.DATASET_NAME,
        config.DATA_SPLIT_SEED,
        model_specs,
        config.TRIPLET_ALPHA
    )

    triplet_metadata_path = triplet_output_dir / "triplet_oof_metadata.pkl"

    triplet_metadata = cast(
        list[dict[str, object]], joblib.load(triplet_metadata_path)
    )

    include_autofeat = any(
        feature_set_requires_autofeat(feature_set)
        for _, feature_set in model_specs
    )

    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    arrays, _ = load_prepared_data(
        prepared_data_dir, include_autofeat=include_autofeat
    )

    evaluated_metadata, triplet_test_predictions = (
        evaluate_triplet_ensembles_on_test(
            triplet_metadata,
            arrays,
            seed=config.RANDOM_SEED,
            elasticnet_max_iter=config.ELASTICNET_MAX_ITER,
            elasticnet_tol=config.ELASTICNET_TOL
        )
    )

    joblib.dump(
        evaluated_metadata,
        triplet_output_dir / "triplet_test_metadata.pkl",
        compress=3
    )
    joblib.dump(
        triplet_test_predictions,
        triplet_output_dir / "triplet_test_predictions.pkl",
        compress=3
    )

    best_by_oof = evaluated_metadata[0]
    best_by_test = min(
        evaluated_metadata, key=lambda item: float(item["test_rmse"])
    )

    print(
        f"{len(evaluated_metadata)} triplet ensembles evaluated on test data."
    )
    print(
        "Triplet test prediction matrix shape: "
        f"{triplet_test_predictions.shape}"
    )
    print(f"Triplet test results saved to: {triplet_output_dir}")
    print()

    print(f"Triplet alpha = {config.TRIPLET_ALPHA:.5f}")
    print(
        f"Best blended OOF triplet: "
        f"OOF RMSE = {float(best_by_oof['oof_rmse']):.5f}, "
        f"test RMSE = {float(best_by_oof['test_rmse']):.5f}, "
        f"columns "
        f"({best_by_oof['i']}, {best_by_oof['j']}, {best_by_oof['k']})"
    )

    print(
        f"Best test triplet among saved {len(evaluated_metadata)} "
        f"ensembles (diagnostic only): "
        f"OOF RMSE = {float(best_by_test['oof_rmse']):.5f}, "
        f"test RMSE = {float(best_by_test['test_rmse']):.5f}, "
        f"columns "
        f"({best_by_test['i']}, {best_by_test['j']}, {best_by_test['k']})"
    )


if __name__ == "__main__":
    main()
