# run_xgboost_optuna.py

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
from tuning.xgboost_optuna import run_optuna_kfold_xgboost


def main() -> None:
    """Run XGBoost Optuna tuning on the selected prepared dataset."""
    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME
    )

    include_autofeat = feature_set_requires_autofeat(
        config.XGBOOST_FEATURE_SET
    )

    arrays, _ = load_prepared_data(
        prepared_data_dir, include_autofeat=include_autofeat
    )

    X_dev, y_dev = get_development_data(arrays, config.XGBOOST_FEATURE_SET)

    (
        study_xgb,
        trial_numbers_xgb,
        oof_predictions_xgb,
        oof_rmses_xgb,
        fold_rmses_xgb,
        hyperparams_xgb,
    ) = run_optuna_kfold_xgboost(
        X_dev,
        y_dev,
        n_trials=config.XGBOOST_N_TRIALS,
        n_jobs=config.XGBOOST_N_JOBS,
        n_splits=config.XGBOOST_N_SPLITS,
        n_estimators_min=config.XGBOOST_N_ESTIMATORS_MIN,
        n_estimators_max=config.XGBOOST_N_ESTIMATORS_MAX,
        n_estimators_step=config.XGBOOST_N_ESTIMATORS_STEP,
        seed=config.RANDOM_SEED,
        verbose=config.XGBOOST_VERBOSE
    )

    print(
        f"\nBest OOF RMSE = {study_xgb.best_value:.5f} "
        f"(trial {study_xgb.best_trial.number})"
    )
    print(f"Best hyperparameters:\n{study_xgb.best_trial.params}")

    result_dir = get_optuna_result_dir(
        config.OPTUNA_RESULTS_DIR,
        config.DATASET_NAME,
        config.XGBOOST_FEATURE_SET,
        "xgboost"
    )

    save_optuna_results(
        result_dir,
        study_xgb,
        trial_numbers_xgb,
        oof_predictions_xgb,
        oof_rmses_xgb,
        fold_rmses_xgb,
        hyperparams_xgb
    )

    print(f"\nXGBoost Optuna results saved to: {result_dir}")


if __name__ == "__main__":
    main()
