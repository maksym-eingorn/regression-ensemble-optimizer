# run_elasticnet_optuna.py

from environment import configure_environment

configure_environment()

import config
from storage import get_dataset_output_dir, load_prepared_data
from tuning.feature_sets import (
    feature_set_requires_autofeat, get_development_data
)
from tuning.result_storage import (
    get_optuna_result_dir, save_optuna_results
)
from tuning.elasticnet_optuna import run_optuna_kfold_elasticnet


def main() -> None:
    """Run ElasticNet Optuna tuning on the selected prepared dataset."""
    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME
    )

    include_autofeat = feature_set_requires_autofeat(
        config.ELASTICNET_FEATURE_SET
    )

    arrays, _ = load_prepared_data(
        prepared_data_dir, include_autofeat=include_autofeat
    )

    X_dev, y_dev = get_development_data(arrays, config.ELASTICNET_FEATURE_SET)

    (
        study_enet,
        trial_numbers_enet,
        oof_predictions_enet,
        oof_rmses_enet,
        fold_rmses_enet,
        hyperparams_enet,
    ) = run_optuna_kfold_elasticnet(
        X_dev,
        y_dev,
        n_trials=config.ELASTICNET_N_TRIALS,
        n_jobs=config.ELASTICNET_N_JOBS,
        n_splits=config.ELASTICNET_N_SPLITS,
        max_iter=config.ELASTICNET_MAX_ITER,
        tol=config.ELASTICNET_TOL,
        seed=config.RANDOM_SEED,
        verbose=config.ELASTICNET_VERBOSE
    )

    print(
        f"\nBest OOF RMSE = {study_enet.best_value:.5f} "
        f"(trial {study_enet.best_trial.number})"
    )
    print(f"Best hyperparameters:\n{study_enet.best_trial.params}")

    result_dir = get_optuna_result_dir(
        config.OPTUNA_RESULTS_DIR,
        config.DATASET_NAME,
        config.ELASTICNET_FEATURE_SET,
        "elasticnet"
    )

    save_optuna_results(
        result_dir,
        study_enet,
        trial_numbers_enet,
        oof_predictions_enet,
        oof_rmses_enet,
        fold_rmses_enet,
        hyperparams_enet
    )

    print(f"\nElasticNet Optuna results saved to: {result_dir}")


if __name__ == "__main__":
    main()
