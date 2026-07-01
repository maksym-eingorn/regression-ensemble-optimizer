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


def find_top_triplet_ensembles(
    P, y, top_n: int = 100, n_threads: int = 1
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
        Number of the best triplets to keep.
    n_threads : int, default=1
        Number of OpenMP threads used by the native search.

    Returns
    -------
    triplet_metadata : list[dict[str, object]]
        Metadata for the best triplet ensembles.
    triplet_oof_predictions : np.ndarray
        Out-of-fold prediction matrix for the best triplet ensembles.
        Shape: (n_samples_dev, n_kept_triplets).
    """
    P, y = _validate_triplet_search_inputs(P, y, top_n, n_threads)

    native = _load_native_triplet_search_module()

    raw_metadata, triplet_oof_predictions = native.find_top_triplets(
        P, y, top_n, n_threads
    )

    triplet_metadata = []

    for rank, item in enumerate(raw_metadata, start=1):
        i, j, k, wi, wj, wk, rmse = item

        triplet_metadata.append({
            "rank": rank,
            "i": int(i),
            "j": int(j),
            "k": int(k),
            "wi": float(wi),
            "wj": float(wj),
            "wk": float(wk),
            "oof_rmse": float(rmse)
        })

    return triplet_metadata, np.asarray(triplet_oof_predictions)
