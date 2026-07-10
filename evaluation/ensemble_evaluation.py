# evaluation/ensemble_evaluation.py

import math
from typing import cast

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

from evaluation.validation import validate_regression_evaluation_inputs
from tuning.feature_sets import get_development_and_test_data


ModelMetadata = dict[str, object]
TripletMetadata = dict[str, object]
GreedyMetadata = dict[str, object]


def _build_model_from_metadata(
    model_metadata: ModelMetadata,
    seed: int,
    elasticnet_max_iter: int,
    elasticnet_tol: float
):
    """Build a regression model from saved base model column metadata."""
    model_name = cast(str, model_metadata["model_name"])
    hyperparams = dict(cast(dict[str, object], model_metadata["hyperparams"]))

    if model_name == "xgboost":
        return xgb.XGBRegressor(
            random_state=seed,
            tree_method="hist",
            n_jobs=1,
            verbosity=0,
            **hyperparams
        )

    if model_name == "lightgbm":
        return lgb.LGBMRegressor(
            random_state=seed,
            feature_fraction_seed=seed,
            bagging_seed=seed,
            extra_seed=seed,
            n_jobs=1,
            verbosity=-1,
            **hyperparams
        )

    if model_name == "elasticnet":
        return ElasticNet(
            max_iter=elasticnet_max_iter,
            tol=elasticnet_tol,
            random_state=seed,
            **hyperparams
        )

    raise ValueError(f"Unsupported model_name in metadata: {model_name}.")


def _predict_base_model_on_test(
    model_metadata: ModelMetadata,
    arrays: dict[str, np.ndarray],
    seed: int,
    elasticnet_max_iter: int,
    elasticnet_tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Retrain one selected base model and predict on the test set."""
    feature_set = cast(str, model_metadata["feature_set"])

    X_dev, y_dev, X_test, y_test = get_development_and_test_data(
        arrays, feature_set
    )

    X_dev, y_dev, X_test, y_test = validate_regression_evaluation_inputs(
        X_dev, y_dev, X_test, y_test
    )

    model = _build_model_from_metadata(
        model_metadata,
        seed=seed,
        elasticnet_max_iter=elasticnet_max_iter,
        elasticnet_tol=elasticnet_tol
    )

    model.fit(X_dev, y_dev)

    y_pred_test = np.asarray(model.predict(X_test), dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)

    return y_pred_test, y_test


def _get_triplet_models(triplet: TripletMetadata) -> list[ModelMetadata]:
    """Return the three base model metadata dictionaries for one triplet."""
    if "models" not in triplet:
        raise KeyError(
            "Triplet metadata does not contain 'models'. "
            "Run run_triplet_search.py before test evaluation."
        )

    models = cast(list[ModelMetadata], triplet["models"])

    if len(models) != 3:
        raise ValueError(
            f"Triplet metadata must contain 3 models, got {len(models)}."
        )

    return models


def _get_triplet_weights(triplet: TripletMetadata) -> np.ndarray:
    """Return the OLS weights for one triplet."""
    return np.array(
        [
            float(triplet["wi"]),
            float(triplet["wj"]),
            float(triplet["wk"])
        ],
        dtype=np.float64
    )


def evaluate_triplet_ensembles_on_test(
    triplet_metadata: list[TripletMetadata],
    arrays: dict[str, np.ndarray],
    seed: int = 42,
    elasticnet_max_iter: int = 20000,
    elasticnet_tol: float = 1e-4
) -> tuple[list[TripletMetadata], np.ndarray]:
    """
    Retrain selected triplet ensembles on the full development set and evaluate
    them on the test set.

    Parameters
    ----------
    triplet_metadata
        Metadata saved by run_triplet_search.py.
    arrays
        Prepared dataset arrays loaded from storage.
    seed
        Random seed used for final model retraining.
    elasticnet_max_iter
        Maximum iterations for final ElasticNet retraining.
    elasticnet_tol
        Optimization tolerance for final ElasticNet retraining.

    Returns
    -------
    evaluated_metadata
        Copy of triplet metadata with added test_rmse values.
    triplet_test_predictions
        Test prediction matrix for evaluated triplets.
        Shape: (n_samples_test, n_kept_triplets).
    """
    if not triplet_metadata:
        raise ValueError("triplet_metadata must contain at least one triplet.")

    prediction_cache: dict[int, np.ndarray] = {}
    y_test_reference = None

    evaluated_metadata = []
    triplet_test_predictions = []

    for triplet in triplet_metadata:
        models = _get_triplet_models(triplet)
        weights = _get_triplet_weights(triplet)

        y_pred_ensemble = None

        for weight, model_metadata in zip(weights, models):
            column_index = int(model_metadata["column_index"])

            if column_index not in prediction_cache:
                y_pred_test, y_test = _predict_base_model_on_test(
                    model_metadata,
                    arrays,
                    seed=seed,
                    elasticnet_max_iter=elasticnet_max_iter,
                    elasticnet_tol=elasticnet_tol
                )

                prediction_cache[column_index] = y_pred_test

                if y_test_reference is None:
                    y_test_reference = y_test
                elif not np.array_equal(y_test_reference, y_test):
                    raise ValueError(
                        "Inconsistent y_test values encountered across feature "
                        "sets."
                    )

            y_pred_test = prediction_cache[column_index]

            if y_pred_ensemble is None:
                y_pred_ensemble = weight * y_pred_test
            else:
                y_pred_ensemble += weight * y_pred_test

        if y_test_reference is None or y_pred_ensemble is None:
            raise RuntimeError("Failed to construct ensemble test predictions.")

        test_rmse = math.sqrt(
            mean_squared_error(y_test_reference, y_pred_ensemble)
        )

        evaluated_triplet = triplet.copy()
        evaluated_triplet["test_rmse"] = float(test_rmse)

        evaluated_metadata.append(evaluated_triplet)
        triplet_test_predictions.append(y_pred_ensemble)

    triplet_test_predictions_matrix = np.column_stack(
        triplet_test_predictions
    ).astype(np.float32)

    return evaluated_metadata, triplet_test_predictions_matrix


def _get_greedy_models(greedy_metadata: GreedyMetadata) -> list[ModelMetadata]:
    """Return selected base model metadata dictionaries for a greedy
    ensemble."""
    if "models" not in greedy_metadata:
        raise KeyError(
            "Greedy metadata does not contain 'models'. "
            "Run run_greedy_ensemble_selection.py before test evaluation."
        )

    models = cast(list[ModelMetadata], greedy_metadata["models"])

    if not models:
        raise ValueError("Greedy metadata must contain at least one model.")

    return models


def _get_greedy_weights(greedy_metadata: GreedyMetadata) -> np.ndarray:
    """Return convex weights for a greedy ensemble."""
    if "weights" not in greedy_metadata:
        raise KeyError("Greedy metadata does not contain 'weights'.")

    weights = np.asarray(greedy_metadata["weights"], dtype=np.float64)

    if weights.ndim != 1:
        raise ValueError(f"Greedy weights must be 1D, got {weights.ndim}D.")

    if len(weights) == 0:
        raise ValueError("Greedy weights must contain at least one value.")

    return weights


def evaluate_greedy_ensemble_on_test(
    greedy_metadata: GreedyMetadata,
    arrays: dict[str, np.ndarray],
    seed: int = 42,
    elasticnet_max_iter: int = 20000,
    elasticnet_tol: float = 1e-4
) -> tuple[GreedyMetadata, np.ndarray]:
    """
    Retrain the selected greedy ensemble on the full development set and
    evaluate it on the test set.

    Parameters
    ----------
    greedy_metadata
        Metadata saved by run_greedy_ensemble_selection.py.
    arrays
        Prepared dataset arrays loaded from storage.
    seed
        Random seed used for final model retraining.
    elasticnet_max_iter
        Maximum iterations for final ElasticNet retraining.
    elasticnet_tol
        Optimization tolerance for final ElasticNet retraining.

    Returns
    -------
    evaluated_metadata
        Copy of greedy metadata with added test_rmse value.
    greedy_test_predictions
        Test prediction matrix for the evaluated greedy ensemble.
        Shape: (n_samples_test, 1).
    """
    models = _get_greedy_models(greedy_metadata)
    weights = _get_greedy_weights(greedy_metadata)

    if len(models) != len(weights):
        raise ValueError(
            f"Number of greedy models and weights must match, "
            f"got {len(models)} != {len(weights)}."
        )

    prediction_cache: dict[int, np.ndarray] = {}
    y_test_reference = None
    y_pred_ensemble = None

    for weight, model_metadata in zip(weights, models):
        column_index = int(model_metadata["column_index"])

        if column_index not in prediction_cache:
            y_pred_test, y_test = _predict_base_model_on_test(
                model_metadata,
                arrays,
                seed=seed,
                elasticnet_max_iter=elasticnet_max_iter,
                elasticnet_tol=elasticnet_tol
            )

            prediction_cache[column_index] = y_pred_test

            if y_test_reference is None:
                y_test_reference = y_test
            elif not np.array_equal(y_test_reference, y_test):
                raise ValueError(
                    "Inconsistent y_test values encountered across feature "
                    "sets."
                )

        y_pred_test = prediction_cache[column_index]

        if y_pred_ensemble is None:
            y_pred_ensemble = weight * y_pred_test
        else:
            y_pred_ensemble += weight * y_pred_test

    if y_test_reference is None or y_pred_ensemble is None:
        raise RuntimeError(
            "Failed to construct greedy ensemble test predictions."
        )

    test_rmse = math.sqrt(mean_squared_error(y_test_reference, y_pred_ensemble))

    evaluated_metadata = greedy_metadata.copy()
    evaluated_metadata["test_rmse"] = float(test_rmse)

    test_predictions_matrix = np.asarray(
        y_pred_ensemble, dtype=np.float32
    ).reshape(-1, 1)

    return evaluated_metadata, test_predictions_matrix
