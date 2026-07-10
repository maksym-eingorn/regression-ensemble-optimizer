# run_greedy_ensemble_selection.py

from environment import configure_environment

configure_environment()

from pathlib import Path
from typing import cast

import joblib
import numpy as np

import config
from ensemble.greedy_selection import find_greedy_ensemble
from ensemble.oof_matrix import build_oof_matrix, load_oof_result_block
from storage import get_dataset_output_dir, load_prepared_data
from tuning.feature_sets import feature_set_requires_autofeat


def _get_model_specs() -> list[tuple[str, str]]:
    """Return model names and configured feature sets for greedy selection."""
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
    """Run Caruana-style greedy ensemble selection on saved out-of-fold
    predictions."""
    model_specs = _get_model_specs()

    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME
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
            feature_set,
            model_name
        )
        for model_name, feature_set in model_specs
    ]

    P, column_metadata = build_oof_matrix(blocks)

    greedy_metadata, greedy_oof_predictions = find_greedy_ensemble(
        P,
        y_dev,
        column_metadata,
        pool_size=config.GREEDY_POOL_SIZE,
        ensemble_size=config.GREEDY_ENSEMBLE_SIZE,
        seed=config.RANDOM_SEED,
        allow_repeats=config.GREEDY_ALLOW_REPEATS
    )

    output_dir = _get_greedy_output_dir(
        config.ENSEMBLE_RESULTS_DIR,
        config.DATASET_NAME,
        model_specs
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        column_metadata,
        output_dir / "base_model_column_metadata.pkl",
        compress=3
    )
    joblib.dump(
        greedy_metadata,
        output_dir / "greedy_oof_metadata.pkl",
        compress=3
    )
    joblib.dump(
        greedy_oof_predictions.reshape(-1, 1).astype(np.float32),
        output_dir / "greedy_oof_predictions.pkl",
        compress=3
    )

    weights = cast(list[float], greedy_metadata["weights"])
    selected_columns = cast(list[int], greedy_metadata["selected_columns"])

    print(f"Unified out-of-fold prediction matrix shape: {P.shape}")
    print(
        "Greedy out-of-fold prediction matrix shape: "
        f"{greedy_oof_predictions.reshape(-1, 1).shape}"
    )

    print(f"Greedy ensemble selection results saved to: {output_dir}")
    print()
    print(
        f"Greedy ensemble OOF RMSE = {greedy_metadata['oof_rmse']:.5f}, "
        "selected columns "
        + "("
        + ", ".join(str(column) for column in selected_columns)
        + ")"
    )
    print(
        "Weights: "
        + "("
        + ", ".join(f"{weight:.5f}" for weight in weights)
        + ")"
    )


if __name__ == "__main__":
    main()
