# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# ensemble/oof_matrix.py

from dataclasses import dataclass
from typing import cast

import numpy as np

from tuning.result_storage import get_optuna_result_dir, load_optuna_results


@dataclass(frozen=True)
class OOFResultBlock:
    """Container for one model family's saved out-of-fold predictions."""
    model_name: str
    feature_set: str
    trial_numbers: np.ndarray
    oof_predictions: np.ndarray
    oof_rmses: np.ndarray
    fold_rmses: np.ndarray
    hyperparams: list[dict[str, object]]


def load_oof_result_block(
    root_dir: str,
    dataset_name: str,
    data_split_seed: int,
    feature_set: str,
    model_name: str
) -> OOFResultBlock:
    """Load one model family's saved Optuna out-of-fold results."""
    result_dir = get_optuna_result_dir(
        root_dir, dataset_name, data_split_seed, feature_set, model_name
    )

    results = load_optuna_results(result_dir)

    hyperparams = cast(list[dict[str, object]], results["hyperparams"])

    return OOFResultBlock(
        model_name=model_name,
        feature_set=feature_set,
        trial_numbers=np.asarray(results["trial_numbers"]),
        oof_predictions=np.asarray(results["oof_predictions"]),
        oof_rmses=np.asarray(results["oof_rmses"]),
        fold_rmses=np.asarray(results["fold_rmses"]),
        hyperparams=hyperparams
    )


def build_oof_matrix(
    blocks: list[OOFResultBlock]
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """
    Build a unified out-of-fold prediction matrix from saved Optuna outputs.

    Parameters
    ----------
    blocks
        Loaded out-of-fold result blocks from model-specific Optuna runs.

    Returns
    -------
    P : np.ndarray
        Unified out-of-fold prediction matrix.
        Shape: (n_samples_dev, n_models).
    column_metadata : list[dict[str, object]]
        Metadata describing which model/trial each column corresponds to.
    """
    if not blocks:
        raise ValueError("At least one out-of-fold result block is required.")

    matrices = []
    column_metadata = []
    n_samples = None
    column_offset = 0

    for block in blocks:
        oof_predictions = np.asarray(block.oof_predictions)

        if oof_predictions.ndim != 2:
            raise ValueError(
                f"Out-of-fold predictions for {block.model_name} must be 2D, "
                f"got {oof_predictions.ndim}D."
            )

        if n_samples is None:
            n_samples = oof_predictions.shape[0]
        elif oof_predictions.shape[0] != n_samples:
            raise ValueError(
                "All out-of-fold prediction matrices must have the same number "
                f"of rows, got {n_samples} and {oof_predictions.shape[0]}."
            )

        n_trials = oof_predictions.shape[1]

        if len(block.trial_numbers) != n_trials:
            raise ValueError(
                f"Trial numbers for {block.model_name} must have length "
                f"{n_trials}, got {len(block.trial_numbers)}."
            )

        if len(block.oof_rmses) != n_trials:
            raise ValueError(
                f"OOF RMSE values for {block.model_name} must have length "
                f"{n_trials}, got {len(block.oof_rmses)}."
            )

        fold_rmses = np.asarray(block.fold_rmses)

        if fold_rmses.ndim != 2:
            raise ValueError(
                f"Fold RMSE values for {block.model_name} must be 2D, "
                f"got {fold_rmses.ndim}D."
            )

        if fold_rmses.shape[0] != n_trials:
            raise ValueError(
                f"Fold RMSE values for {block.model_name} must have "
                f"{n_trials} rows, got {fold_rmses.shape[0]}."
            )

        if len(block.hyperparams) != n_trials:
            raise ValueError(
                f"Hyperparameter entries for {block.model_name} must have "
                f"length {n_trials}, got {len(block.hyperparams)}."
            )

        for local_column in range(n_trials):
            column_metadata.append({
                "column_index": column_offset + local_column,
                "model_name": block.model_name,
                "feature_set": block.feature_set,
                "trial_number": int(block.trial_numbers[local_column]),
                "oof_rmse": float(block.oof_rmses[local_column]),
                "fold_rmses": fold_rmses[local_column].astype(float).tolist(),
                "hyperparams": block.hyperparams[local_column]
            })

        matrices.append(oof_predictions)
        column_offset += n_trials

    P = np.hstack(matrices).astype(np.float64)

    return P, column_metadata
