# evaluation/elasticnet_evaluation.py

import math

import optuna
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

from evaluation.validation import validate_regression_evaluation_inputs


def evaluate_best_elasticnet_on_test(
    study_enet: optuna.study.Study,
    X_dev,
    y_dev,
    X_test,
    y_test,
    max_iter: int = 20000,
    tol: float = 1e-4,
    seed: int = 42
) -> float:
    """
    Retrain the best ElasticNet model from Optuna and evaluate it on the test
    set.

    Parameters
    ----------
    study_enet : optuna.study.Study
        Completed Optuna study for ElasticNet.
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
    max_iter : int, default=20000
        Maximum number of ElasticNet optimization iterations.
    tol : float, default=1e-4
        Optimization tolerance for ElasticNet.
    seed : int, default=42
        Random seed used by the final ElasticNet model.

    Returns
    -------
    test_rmse : float
        RMSE of the best ElasticNet model evaluated on the test set.
    """
    X_dev, y_dev, X_test, y_test = validate_regression_evaluation_inputs(
        X_dev, y_dev, X_test, y_test
    )

    best_params = study_enet.best_params.copy()

    model = ElasticNet(
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
        **best_params
    )

    model.fit(X_dev, y_dev)

    y_pred_test = model.predict(X_test)
    test_rmse = math.sqrt(mean_squared_error(y_test, y_pred_test))

    return test_rmse
