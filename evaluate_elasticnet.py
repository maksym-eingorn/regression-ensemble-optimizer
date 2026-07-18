# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# evaluate_elasticnet.py

from environment import configure_environment

configure_environment()

from typing import cast

import optuna

import config
from evaluation.elasticnet_evaluation import evaluate_best_elasticnet_on_test
from storage import get_dataset_output_dir, load_prepared_data
from tuning.feature_sets import (
    feature_set_requires_autofeat, get_development_and_test_data
)
from tuning.result_storage import get_optuna_result_dir, load_optuna_results


def main() -> None:
    """Evaluate the best tuned ElasticNet model on the test set."""
    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    include_autofeat = feature_set_requires_autofeat(
        config.ELASTICNET_FEATURE_SET
    )

    arrays, _ = load_prepared_data(
        prepared_data_dir, include_autofeat=include_autofeat
    )

    X_dev, y_dev, X_test, y_test = get_development_and_test_data(
        arrays, config.ELASTICNET_FEATURE_SET
    )

    result_dir = get_optuna_result_dir(
        config.OPTUNA_RESULTS_DIR,
        config.DATASET_NAME,
        config.DATA_SPLIT_SEED,
        config.ELASTICNET_FEATURE_SET,
        "elasticnet"
    )

    optuna_results = load_optuna_results(result_dir)
    study_enet = cast(optuna.study.Study, optuna_results["study"])

    test_rmse_enet = evaluate_best_elasticnet_on_test(
        study_enet,
        X_dev,
        y_dev,
        X_test,
        y_test,
        max_iter=config.ELASTICNET_MAX_ITER,
        tol=config.ELASTICNET_TOL,
        seed=config.RANDOM_SEED
    )

    print(f"Best ElasticNet model: test RMSE = {test_rmse_enet:.5f}")


if __name__ == "__main__":
    main()
