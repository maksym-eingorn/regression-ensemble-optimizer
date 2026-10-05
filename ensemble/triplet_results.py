# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# ensemble/triplet_results.py

import math
from decimal import Decimal
from pathlib import Path

from storage import get_data_split_name


def validate_triplet_alpha(alpha: float) -> float:
    """Validate and return a triplet OLS-to-equal blending coefficient."""
    if isinstance(alpha, bool):
        raise ValueError(
            f"alpha must be a finite float from 0 to 1, got {alpha}."
        )

    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"alpha must be a finite float from 0 to 1, got {alpha}."
        ) from exc

    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            f"alpha must be a finite float from 0 to 1, got {alpha}."
        )

    return 0.0 if value == 0.0 else value


def get_triplet_alpha_folder_name(alpha: float) -> str:
    """Return the alpha-specific triplet result folder name."""
    value = validate_triplet_alpha(alpha)

    alpha_text = format(Decimal(str(value)).normalize(), "f")

    return f"triplet_alpha_{alpha_text}"


def get_triplet_output_dir(
    root_dir: str | Path,
    dataset_name: str,
    data_split_seed: int,
    model_specs: list[tuple[str, str]],
    alpha: float
) -> Path:
    """Return the alpha-specific triplet ensemble result directory."""
    spec_name = "__".join(
        f"{model_name}-{feature_set}"
        for model_name, feature_set in model_specs
    )

    return (
        Path(root_dir)
        / dataset_name
        / get_data_split_name(data_split_seed)
        / spec_name
        / get_triplet_alpha_folder_name(alpha)
    )
