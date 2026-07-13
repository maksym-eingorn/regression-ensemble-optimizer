# evaluate_lightgbm.py

from environment import configure_environment

configure_environment()

from typing import cast

import optuna

import config
from evaluation.lightgbm_evaluation import evaluate_best_lightgbm_on_test
from storage import get_dataset_output_dir, load_prepared_data
from tuning.feature_sets import (
    feature_set_requires_autofeat, get_development_and_test_data
)
from tuning.result_storage import get_optuna_result_dir, load_optuna_results


def main() -> None:
    """Evaluate the best tuned LightGBM model on the test set."""
    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    include_autofeat = feature_set_requires_autofeat(
        config.LIGHTGBM_FEATURE_SET
    )

    arrays, _ = load_prepared_data(
        prepared_data_dir, include_autofeat=include_autofeat
    )

    X_dev, y_dev, X_test, y_test = get_development_and_test_data(
        arrays, config.LIGHTGBM_FEATURE_SET
    )

    result_dir = get_optuna_result_dir(
        config.OPTUNA_RESULTS_DIR,
        config.DATASET_NAME,
        config.DATA_SPLIT_SEED,
        config.LIGHTGBM_FEATURE_SET,
        "lightgbm"
    )

    optuna_results = load_optuna_results(result_dir)
    study_lgb = cast(optuna.study.Study, optuna_results["study"])

    test_rmse_lgb = evaluate_best_lightgbm_on_test(
        study_lgb,
        X_dev,
        y_dev,
        X_test,
        y_test,
        seed=config.RANDOM_SEED
    )

    print(f"Best LightGBM model: test RMSE = {test_rmse_lgb:.5f}")


if __name__ == "__main__":
    main()
