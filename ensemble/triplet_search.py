# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# ensemble/triplet_search.py

import numpy as np

from ensemble.triplet_results import validate_triplet_alpha


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

    if P.shape[0] < 1:
        raise ValueError(
            f"Triplet search requires at least 1 sample, got {P.shape[0]}."
        )

    if P.shape[1] < 3:
        raise ValueError(
            f"Triplet search requires at least 3 model columns, "
            f"got {P.shape[1]}."
        )

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError(f"top_n must be a positive integer, got {top_n}.")

    if (
        isinstance(n_threads, bool)
        or not isinstance(n_threads, int)
        or n_threads < 1
    ):
        raise ValueError(
            f"n_threads must be a positive integer, got {n_threads}."
        )

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
    weight_l1_limit: float | None = None,
    alpha: float = 0.0
) -> tuple[list[dict[str, object]], np.ndarray]:
    """
    Run exact exhaustive alpha-blended triplet search using the compiled C++
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
        Maximum number of top OOF-ranked triplets returned by the native
        exhaustive search. If weight_l1_limit is not None, only triplets
        satisfying the L1 weight guard are eligible for top-N retention. Fewer
        than top_n triplets may be returned only if fewer valid triplets exist.
    n_threads : int, default=1
        Number of OpenMP threads used by the native search.
    weight_l1_limit : float | None, default=None
        Optional maximum allowed L1 norm of the three final alpha-blended
        ensemble weights, defined as abs(wi) + abs(wj) + abs(wk). If None,
        no weight guard is applied. If a float is provided, the native
        exhaustive search rejects triplets whose final weight L1 norm exceeds
        this value before they can enter the top-N candidate heaps. This is
        intended to reject triplets with unusually strong positive-negative
        weight cancellation.
    alpha : float, default=0.0
        Fixed shrinkage coefficient used to blend each triplet's unrestricted
        OLS weights with equal weights. Must be between 0 and 1. An alpha of
        0 uses pure OLS weights, while an alpha of 1 uses equal weights
        (1/3, 1/3, 1/3).

    Returns
    -------
    triplet_metadata : list[dict[str, object]]
        Metadata for the returned triplet ensembles, including the final
        alpha-blended weights, unrestricted OLS weights, alpha-blended OOF
        RMSE, and unrestricted OLS OOF RMSE. If weight_l1_limit is not None,
        all returned triplets satisfy the L1 weight guard. The reported rank is
        the OOF rank among the valid triplets returned by the native exhaustive
        search.
    triplet_oof_predictions : np.ndarray
        Out-of-fold prediction matrix produced using the final alpha-blended
        weights for the kept triplet ensembles.
        Shape: (n_samples_dev, n_returned_triplets).
    """
    P, y = _validate_triplet_search_inputs(P, y, top_n, n_threads)
    weight_l1_limit = _validate_weight_l1_limit(weight_l1_limit)
    alpha = validate_triplet_alpha(alpha)

    native = _load_native_triplet_search_module()

    native_weight_l1_limit = (
        float("inf") if weight_l1_limit is None else weight_l1_limit
    )

    raw_metadata, triplet_oof_predictions = native.find_top_triplets(
        P, y, top_n, n_threads, native_weight_l1_limit, alpha
    )

    triplet_oof_predictions = np.asarray(triplet_oof_predictions)

    triplet_metadata = []

    for rank, item in enumerate(raw_metadata, start=1):
        i, j, k, wi, wj, wk, ols_wi, ols_wj, ols_wk, rmse, ols_rmse = item

        weight_l1 = abs(float(wi)) + abs(float(wj)) + abs(float(wk))

        ols_weight_l1 = (
            abs(float(ols_wi)) + abs(float(ols_wj)) + abs(float(ols_wk))
        )

        triplet_metadata.append({
            "method": "ols_equal_weight_blend",
            "alpha": float(alpha),
            "rank": rank,
            "i": int(i),
            "j": int(j),
            "k": int(k),
            "wi": float(wi),
            "wj": float(wj),
            "wk": float(wk),
            "weight_l1": float(weight_l1),
            "ols_wi": float(ols_wi),
            "ols_wj": float(ols_wj),
            "ols_wk": float(ols_wk),
            "ols_weight_l1": float(ols_weight_l1),
            "oof_rmse": float(rmse),
            "ols_oof_rmse": float(ols_rmse)
        })

    return triplet_metadata, triplet_oof_predictions
