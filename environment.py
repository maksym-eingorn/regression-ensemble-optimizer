# environment.py

import os


def configure_environment() -> None:
    """
    Limit hidden parallelism in numerical libraries to improve reproducibility.

    This should be called before importing NumPy, scikit-learn, AutoFeat,
    XGBoost, LightGBM, or other numerical libraries.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("XGB_NUM_THREADS", "1")
    os.environ.setdefault("LGBM_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
