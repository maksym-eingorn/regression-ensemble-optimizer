# setup_triplet_search.py

import os
import sys
from pathlib import Path

import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext


eigen_path = Path(
    os.environ.get("EIGEN_INCLUDE_DIR", r"C:\Libraries\eigen-3.4.1")
)

if not eigen_path.exists():
    raise RuntimeError(
        "Eigen directory was not found. "
        "Set EIGEN_INCLUDE_DIR to the Eigen folder, for example: "
        r"C:\Libraries\eigen-3.4.1"
    )

if sys.platform == "win32":
    define_macros = [("NOMINMAX", "1")]
    extra_compile_args = ["/O2", "/openmp", "/MD", "/EHsc"]
    extra_link_args = []
else:
    define_macros = []
    extra_compile_args = ["-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "ensemble._triplet_search",
        ["native/triplet_search.cpp"],
        include_dirs=[str(eigen_path)],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=17
    )
]

setuptools.setup(
    name="regression_ensemble_optimizer_triplet_search",
    version="0.1",
    packages=["ensemble"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext}
)
