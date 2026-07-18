# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# tuning/result_storage.py

from pathlib import Path

import joblib
import numpy as np
import optuna

from storage import get_dataset_output_dir


def get_optuna_result_dir(
    root_dir: str | Path,
    dataset_name: str,
    data_split_seed: int,
    feature_set: str,
    model_name: str
) -> Path:
    """Return the output directory for Optuna results."""
    return (
        get_dataset_output_dir(root_dir, dataset_name, data_split_seed)
        / feature_set
        / model_name
    )


def save_optuna_results(
    output_dir: str | Path,
    study: optuna.study.Study,
    trial_numbers: np.ndarray,
    oof_predictions: np.ndarray,
    oof_rmses: np.ndarray,
    fold_rmses: np.ndarray,
    hyperparams: list[dict[str, object]]
) -> None:
    """Save Optuna study results and out-of-fold prediction artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(study, output_path / "study.pkl", compress=3)
    joblib.dump(trial_numbers, output_path / "trial_numbers.pkl", compress=3)
    joblib.dump(
        oof_predictions, output_path / "oof_predictions.pkl", compress=3
    )
    joblib.dump(oof_rmses, output_path / "oof_rmses.pkl", compress=3)
    joblib.dump(fold_rmses, output_path / "fold_rmses.pkl", compress=3)
    joblib.dump(hyperparams, output_path / "hyperparams.pkl", compress=3)


def load_optuna_results(output_dir: str | Path) -> dict[str, object]:
    """Load saved Optuna study results and out-of-fold prediction artifacts."""
    output_path = Path(output_dir)

    return {
        "study": joblib.load(output_path / "study.pkl"),
        "trial_numbers": joblib.load(output_path / "trial_numbers.pkl"),
        "oof_predictions": joblib.load(output_path / "oof_predictions.pkl"),
        "oof_rmses": joblib.load(output_path / "oof_rmses.pkl"),
        "fold_rmses": joblib.load(output_path / "fold_rmses.pkl"),
        "hyperparams": joblib.load(output_path / "hyperparams.pkl")
    }
