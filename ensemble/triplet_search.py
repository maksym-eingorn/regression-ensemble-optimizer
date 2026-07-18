# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# ensemble/triplet_search.py

import numpy as np


def _load_native_triplet_search_module():
    """Load the compiled C++ triplet search extension."""
    try:
        from ensemble import _triplet_search
    except ImportError as exc:
        raise ImportError(
            "The compiled C++ triplet search extension was not found. "
            "Build it first with: "
            "python setup_triplet_search.py build_ext --inplace"
        ) from exc

    return _triplet_search


def _validate_triplet_search_inputs(
    P, y, top_n: int, n_threads: int
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert inputs for native triplet search."""
    P = np.ascontiguousarray(P, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)

    if P.ndim != 2:
        raise ValueError(
            f"P must be 2D (n_samples_dev, n_models), got {P.ndim}D."
        )

    if y.ndim != 1:
        raise ValueError(f"y must be 1D (n_samples_dev,), got {y.ndim}D.")

    if P.shape[0] != y.shape[0]:
        raise ValueError(
            f"Numbers of samples in P and y must match, "
            f"got {P.shape[0]} != {y.shape[0]}."
        )

    if P.shape[1] < 3:
        raise ValueError(
            f"Triplet search requires at least 3 model columns, "
            f"got {P.shape[1]}."
        )

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError(f"top_n must be a positive integer, got {top_n}.")

    if not isinstance(n_threads, int):
        raise ValueError(f"n_threads must be an integer, got {n_threads}.")

    return P, y


def _validate_weight_l1_limit(
    weight_l1_limit: float | None
) -> float | None:
    """Validate the optional triplet weight L1 limit."""
    if weight_l1_limit is None:
        return None

    if isinstance(weight_l1_limit, bool):
        raise ValueError(
            "weight_l1_limit must be a positive float or None, "
            f"got {weight_l1_limit}."
        )

    try:
        value = float(weight_l1_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "weight_l1_limit must be a positive float or None, "
            f"got {weight_l1_limit}."
        ) from exc

    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            "weight_l1_limit must be a positive finite float or None, "
            f"got {weight_l1_limit}."
        )

    return value


def find_top_triplet_ensembles(
    P,
    y,
    top_n: int = 100,
    n_threads: int = 1,
    weight_l1_limit: float | None = None
) -> tuple[list[dict[str, object]], np.ndarray]:
    """
    Run exact exhaustive OLS-weighted triplet search using the compiled C++
    extension.

    Parameters
    ----------
    P
        Unified out-of-fold prediction matrix.
        Shape: (n_samples_dev, n_models).
    y
        Development target vector.
        Shape: (n_samples_dev,).
    top_n : int, default=100
        Number of top unguarded triplets requested from the native exhaustive
        search before optional Python-side filtering. If weight_l1_limit is not
        None, the final number of kept triplets may be smaller.
    n_threads : int, default=1
        Number of OpenMP threads used by the native search.
    weight_l1_limit : float | None, default=None
        Optional maximum allowed L1 norm of the three OLS ensemble weights,
        defined as abs(wi) + abs(wj) + abs(wk). If None, no weight guard is
        applied. If a float is provided, triplets whose weight L1 norm exceeds
        this value are discarded after the native exhaustive search returns its
        top candidates. This is intended to reject triplets with unusually
        strong positive-negative weight cancellation.

    Returns
    -------
    triplet_metadata : list[dict[str, object]]
        Metadata for the kept triplet ensembles. If weight_l1_limit is not
        None, the list contains only triplets passing the L1 weight guard.
        The reported rank is the post-filtered rank, while unfiltered_rank
        gives the original rank returned by the native exhaustive search.
    triplet_oof_predictions : np.ndarray
        Out-of-fold prediction matrix for the kept triplet ensembles.
        Shape: (n_samples_dev, n_kept_triplets).
    """
    P, y = _validate_triplet_search_inputs(P, y, top_n, n_threads)
    weight_l1_limit = _validate_weight_l1_limit(weight_l1_limit)

    native = _load_native_triplet_search_module()

    raw_metadata, triplet_oof_predictions = native.find_top_triplets(
        P, y, top_n, n_threads
    )

    triplet_oof_predictions = np.asarray(triplet_oof_predictions)

    triplet_metadata = []
    kept_columns = []

    for unfiltered_rank, item in enumerate(raw_metadata, start=1):
        i, j, k, wi, wj, wk, rmse = item

        weight_l1 = abs(float(wi)) + abs(float(wj)) + abs(float(wk))

        if weight_l1_limit is not None and weight_l1 > weight_l1_limit:
            continue

        kept_columns.append(unfiltered_rank - 1)

        triplet_metadata.append({
            "rank": len(triplet_metadata) + 1,
            "unfiltered_rank": unfiltered_rank,
            "i": int(i),
            "j": int(j),
            "k": int(k),
            "wi": float(wi),
            "wj": float(wj),
            "wk": float(wk),
            "weight_l1": float(weight_l1),
            "oof_rmse": float(rmse)
        })

    if kept_columns:
        filtered_predictions = triplet_oof_predictions[:, kept_columns]
    else:
        filtered_predictions = np.empty((P.shape[0], 0), dtype=np.float64)

    return triplet_metadata, filtered_predictions
