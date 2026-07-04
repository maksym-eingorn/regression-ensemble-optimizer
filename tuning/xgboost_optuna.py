# tuning/xgboost_optuna.py

import math

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from tuning.validation import (
    validate_integer_range_settings,
    validate_kfold_settings,
    validate_positive_integer,
    validate_regression_inputs
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna_kfold_xgboost(
    X_dev,
    y_dev,
    n_trials: int = 1000,
    n_jobs: int = 1,
    n_splits: int = 5,
    n_estimators_min: int = 100,
    n_estimators_max: int = 1000,
    n_estimators_step: int = 50,
    seed: int = 42,
    verbose: bool = False
) -> tuple[
    optuna.study.Study,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, object]]
]:
    """
    Run Optuna to tune an XGBoost regressor with K-fold cross-validation.

    Out-of-fold predictions are stored for future ensemble search. Each Optuna
    trial trains one XGBoost configuration across all K folds and produces one
    full out-of-fold prediction vector for the development set.

    Parameters
    ----------
    X_dev
        Development feature matrix used for model training and cross-validation.
        Expected shape: (n_samples_dev, n_features).
    y_dev
        Development target vector corresponding to X_dev.
        Expected shape: (n_samples_dev,).
    n_trials : int, default=1000
        Number of Optuna trials.
    n_jobs : int, default=1
        Number of parallel Optuna workers. Sequential execution is recommended
        for maximum reproducibility.
    n_splits : int, default=5
        Number of K-fold splits used for cross-validation.
    n_estimators_min : int, default=100
        Minimum number of estimators considered by Optuna.
    n_estimators_max : int, default=1000
        Maximum number of estimators considered by Optuna.
    n_estimators_step : int, default=50
        Step size for the Optuna integer search over n_estimators.
    seed : int, default=42
        Random seed used by the Optuna sampler, K-fold splitting, and XGBoost.
    verbose : bool, default=False
        Whether to print the OOF RMSE for each trial.

    Returns
    -------
    study : optuna.study.Study
        Completed Optuna study.
    trial_numbers : np.ndarray
        Trial numbers sorted in ascending order.
        Shape: (n_trials,).
    oof_predictions : np.ndarray
        Out-of-fold prediction matrix.
        Shape: (n_samples_dev, n_trials). Each column corresponds to one trial.
    oof_rmses : np.ndarray
        OOF RMSE values for each trial.
        Shape: (n_trials,).
    fold_rmses : np.ndarray
        Per-fold RMSE values for each trial.
        Shape: (n_trials, n_splits).
    hyperparams : list[dict[str, object]]
        Hyperparameter dictionary for each trial, ordered by trial number.
    """
    X_dev, y_dev = validate_regression_inputs(X_dev, y_dev)

    validate_positive_integer(n_trials, "n_trials")
    validate_positive_integer(n_jobs, "n_jobs")
    validate_kfold_settings(n_splits, X_dev.shape[0])
    validate_integer_range_settings(
        n_estimators_min,
        n_estimators_max,
        n_estimators_step,
        "n_estimators"
    )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    results = []

    def objective_xgb(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                n_estimators_min,
                n_estimators_max,
                step=n_estimators_step
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 2, 15
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.5, 1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "colsample_bynode": trial.suggest_float(
                "colsample_bynode", 0.6, 1.0
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 0.01, 50.0, log=True
            ),
            "gamma": trial.suggest_float(
                "gamma", 0.0, 10.0
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.001, 30.0, log=True
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 0.001, 30.0, log=True
            )
        }

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        oof_pred = np.zeros(len(y_dev), dtype=np.float32)
        per_fold_rmses = []

        for train_idx, valid_idx in kf.split(X_dev):
            X_train, X_valid = X_dev[train_idx], X_dev[valid_idx]
            y_train, y_valid = y_dev[train_idx], y_dev[valid_idx]

            model = xgb.XGBRegressor(
                random_state=seed,
                tree_method="hist",
                n_jobs=1,
                verbosity=0,
                **params
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)

            oof_pred[valid_idx] = y_pred

            fold_rmse = math.sqrt(mean_squared_error(y_valid, y_pred))
            per_fold_rmses.append(fold_rmse)

        fold_rmses_array = np.array(per_fold_rmses, dtype=float)
        oof_rmse = math.sqrt(mean_squared_error(y_dev, oof_pred))

        results.append((
            trial.number,
            oof_pred.copy(),
            oof_rmse,
            fold_rmses_array,
            params
        ))

        if verbose:
            print(f"Trial {trial.number} | OOF RMSE = {oof_rmse:.5f}")

        return oof_rmse

    study.optimize(objective_xgb, n_trials=n_trials, n_jobs=n_jobs)

    results.sort(key=lambda item: item[0])

    trial_numbers = np.array([item[0] for item in results], dtype=int)

    oof_predictions = np.column_stack(
        [item[1] for item in results]
    ).astype(np.float32)

    oof_rmses = np.array([item[2] for item in results], dtype=float)

    fold_rmses = np.vstack([item[3] for item in results]).astype(float)

    hyperparams = [item[4] for item in results]

    return (
        study,
        trial_numbers,
        oof_predictions,
        oof_rmses,
        fold_rmses,
        hyperparams
    )
