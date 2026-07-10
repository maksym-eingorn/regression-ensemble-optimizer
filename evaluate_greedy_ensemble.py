# evaluate_greedy_ensemble.py

from environment import configure_environment

configure_environment()

from pathlib import Path
from typing import cast

import joblib

import config
from evaluation.ensemble_evaluation import evaluate_greedy_ensemble_on_test
from storage import get_dataset_output_dir, load_prepared_data
from tuning.feature_sets import feature_set_requires_autofeat


def _get_model_specs() -> list[tuple[str, str]]:
    """Return model names and configured feature sets for greedy evaluation."""
    return [
        ("xgboost", config.XGBOOST_FEATURE_SET),
        ("lightgbm", config.LIGHTGBM_FEATURE_SET),
        ("elasticnet", config.ELASTICNET_FEATURE_SET)
    ]


def _get_greedy_output_dir(
    root_dir: str, dataset_name: str, model_specs: list[tuple[str, str]]
) -> Path:
    """Return the output directory for greedy ensemble selection results."""
    spec_name = "__".join(
        f"{model_name}-{feature_set}"
        for model_name, feature_set in model_specs
    )

    return Path(root_dir) / dataset_name / spec_name / "greedy_selection"


def main() -> None:
    """Evaluate the saved greedy ensemble on the held-out test set."""
    model_specs = _get_model_specs()

    greedy_output_dir = _get_greedy_output_dir(
        config.ENSEMBLE_RESULTS_DIR,
        config.DATASET_NAME,
        model_specs
    )

    greedy_metadata_path = greedy_output_dir / "greedy_oof_metadata.pkl"

    greedy_metadata = cast(
        dict[str, object], joblib.load(greedy_metadata_path)
    )

    include_autofeat = any(
        feature_set_requires_autofeat(feature_set)
        for _, feature_set in model_specs
    )

    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME
    )

    arrays, _ = load_prepared_data(
        prepared_data_dir, include_autofeat=include_autofeat
    )

    evaluated_metadata, greedy_test_predictions = (
        evaluate_greedy_ensemble_on_test(
            greedy_metadata,
            arrays,
            seed=config.RANDOM_SEED,
            elasticnet_max_iter=config.ELASTICNET_MAX_ITER,
            elasticnet_tol=config.ELASTICNET_TOL
        )
    )

    joblib.dump(
        evaluated_metadata,
        greedy_output_dir / "greedy_test_metadata.pkl",
        compress=3
    )
    joblib.dump(
        greedy_test_predictions,
        greedy_output_dir / "greedy_test_predictions.pkl",
        compress=3
    )

    print("Greedy ensemble evaluated on test data.")
    print(
        "Greedy test prediction matrix shape: "
        f"{greedy_test_predictions.shape}"
    )
    print(f"Greedy test results saved to: {greedy_output_dir}")
    print()

    print(
        f"Greedy ensemble: "
        f"OOF RMSE = {float(evaluated_metadata['oof_rmse']):.5f}, "
        f"test RMSE = {float(evaluated_metadata['test_rmse']):.5f}"
    )


if __name__ == "__main__":
    main()
