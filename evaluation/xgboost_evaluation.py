# evaluation/xgboost_evaluation.py

import math

import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from evaluation.validation import validate_regression_evaluation_inputs


def evaluate_best_xgboost_on_test(
    study_xgb: optuna.study.Study,
    X_dev,
    y_dev,
    X_test,
    y_test,
    seed: int = 42
) -> float:
    """
    Retrain the best XGBoost model from Optuna and evaluate it on the test set.

    Parameters
    ----------
    study_xgb : optuna.study.Study
        Completed Optuna study for XGBoost.
    X_dev
        Development feature matrix used for final model retraining.
        Expected shape: (n_samples_dev, n_features).
    y_dev
        Development target vector corresponding to X_dev.
        Expected shape: (n_samples_dev,).
    X_test
        Test feature matrix used for evaluation.
        Expected shape: (n_samples_test, n_features).
    y_test
        Test target vector corresponding to X_test.
        Expected shape: (n_samples_test,).
    seed : int, default=42
        Random seed used by the final XGBoost model.

    Returns
    -------
    test_rmse : float
        RMSE of the best XGBoost model evaluated on the test set.
    """
    X_dev, y_dev, X_test, y_test = validate_regression_evaluation_inputs(
        X_dev, y_dev, X_test, y_test
    )

    best_params = study_xgb.best_params.copy()

    model = xgb.XGBRegressor(
        random_state=seed,
        tree_method="hist",
        n_jobs=1,
        verbosity=0,
        **best_params
    )

    model.fit(X_dev, y_dev)

    y_pred_test = model.predict(X_test)
    test_rmse = math.sqrt(mean_squared_error(y_test, y_pred_test))

    return test_rmse
