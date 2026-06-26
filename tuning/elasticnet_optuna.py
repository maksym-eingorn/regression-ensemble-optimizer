# tuning/elasticnet_optuna.py

import math

import numpy as np
import optuna
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from tuning.validation import (
    validate_kfold_settings,
    validate_positive_integer,
    validate_regression_inputs
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna_kfold_elasticnet(
    X_dev,
    y_dev,
    n_trials: int = 100,
    n_jobs: int = 1,
    n_splits: int = 5,
    max_iter: int = 20000,
    tol: float = 1e-4,
    seed: int = 42,
    verbose: bool = False
) -> tuple[
    optuna.study.Study,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, object]]
]:
    """
    Run Optuna to tune an ElasticNet regressor with K-fold cross-validation.

    Out-of-fold predictions are stored for future ensemble search. Each Optuna
    trial trains one ElasticNet configuration across all K folds and produces
    one full out-of-fold prediction vector for the development set.

    Parameters
    ----------
    X_dev
        Development feature matrix used for model training and cross-validation.
        Expected shape: (n_samples_dev, n_features).
    y_dev
        Development target vector corresponding to X_dev.
        Expected shape: (n_samples_dev,).
    n_trials : int, default=100
        Number of Optuna trials.
    n_jobs : int, default=1
        Number of parallel Optuna workers. Sequential execution is recommended
        for maximum reproducibility.
    n_splits : int, default=5
        Number of K-fold splits used for cross-validation.
    max_iter : int, default=20000
        Maximum number of ElasticNet optimization iterations.
    tol : float, default=1e-4
        Optimization tolerance for ElasticNet.
    seed : int, default=42
        Random seed used by the Optuna sampler, K-fold splitting, and
        ElasticNet.
    verbose : bool, default=False
        Whether to print the mean cross-validation RMSE for each trial.

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
    rmses : np.ndarray
        Mean cross-validation RMSE for each trial.
        Shape: (n_trials,).
    hyperparams : list[dict[str, object]]
        Hyperparameter dictionary for each trial, ordered by trial number.
    """
    X_dev, y_dev = validate_regression_inputs(X_dev, y_dev)

    validate_positive_integer(n_trials, "n_trials")
    validate_positive_integer(n_jobs, "n_jobs")
    validate_kfold_settings(n_splits, X_dev.shape[0])
    validate_positive_integer(max_iter, "max_iter")

    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}.")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    results = []

    def objective_enet(trial: optuna.trial.Trial) -> float:
        params = {
            "alpha": trial.suggest_float(
                "alpha", 1e-4, 10.0, log=True
            ),
            "l1_ratio": trial.suggest_float(
                "l1_ratio", 0.001, 1.0
            )
        }

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        oof_pred = np.zeros(len(y_dev), dtype=np.float32)
        fold_rmses = []

        for train_idx, valid_idx in kf.split(X_dev):
            X_train, X_valid = X_dev[train_idx], X_dev[valid_idx]
            y_train, y_valid = y_dev[train_idx], y_dev[valid_idx]

            model = ElasticNet(
                max_iter=max_iter,
                tol=tol,
                random_state=seed,
                **params
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)

            oof_pred[valid_idx] = y_pred

            fold_rmse = math.sqrt(mean_squared_error(y_valid, y_pred))
            fold_rmses.append(fold_rmse)

        mean_rmse = float(np.mean(fold_rmses))

        results.append((trial.number, oof_pred.copy(), mean_rmse, params))

        if verbose:
            print(f"Trial {trial.number} | mean CV RMSE = {mean_rmse:.5f}")

        return mean_rmse

    study.optimize(objective_enet, n_trials=n_trials, n_jobs=n_jobs)

    results.sort(key=lambda item: item[0])

    trial_numbers = np.array([item[0] for item in results], dtype=int)

    oof_predictions = np.column_stack(
        [item[1] for item in results]
    ).astype(np.float32)

    rmses = np.array([item[2] for item in results], dtype=float)

    hyperparams = [item[3] for item in results]

    return study, trial_numbers, oof_predictions, rmses, hyperparams
