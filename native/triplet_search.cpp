// Copyright 2026 Maksym Eingorn
// SPDX-License-Identifier: Apache-2.0

// native/triplet_search.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <omp.h>
#include <tuple>
#include <cmath>
#include <vector>
#include <algorithm>

namespace py = pybind11;

// RAII guard: temporarily set OpenMP threads and restore on scope exit
struct OmpThreadGuard {
    int prev_threads;
    bool active;

    explicit OmpThreadGuard(int n_threads)
        : prev_threads(omp_get_max_threads()), active(false) {
        if (n_threads > 0) {
            omp_set_num_threads(n_threads);
            active = true;
        }
    }

    ~OmpThreadGuard() {
        if (active) {
            omp_set_num_threads(prev_threads);
        }
    }
};


struct TripletCandidate {
    double rmse;
    int i, j, k;
    double wi, wj, wk;

    // For max-heap: the worst candidate should be on top
    bool operator<(const TripletCandidate& other) const {
        return rmse < other.rmse;
    }
};

std::tuple<
    std::vector<std::tuple<int,int,int,double,double,double,double>>,
    py::array_t<double>
> find_top_triplets(
    py::array_t<double> P_in,
    py::array_t<double> y_in,
    int top_n,
    int n_threads = -1  // if > 0, temporarily set OpenMP threads for this call
) {
    if (P_in.ndim() != 2) {
        throw py::value_error("P_in must be 2D.");
    }

    if (y_in.ndim() != 1) {
        throw py::value_error("y_in must be 1D.");
    }

    if (P_in.shape(0) != y_in.shape(0)) {
        throw py::value_error(
            "P_in and y_in must have the same number of samples."
        );
    }

    if (P_in.shape(1) < 3) {
        throw py::value_error(
            "Triplet search requires at least 3 model columns."
        );
    }

    if (top_n < 1) {
        throw py::value_error("top_n must be a positive integer.");
    }

    OmpThreadGuard guard(n_threads);

    auto P = P_in.unchecked<2>();
    auto y = y_in.unchecked<1>();

    const int n_samples = static_cast<int>(P.shape(0));
    const int M = static_cast<int>(P.shape(1));

    // Compute Gram matrix G = P^T P
    Eigen::MatrixXd G(M, M);
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = i; j < M; j++) {
            double sum = 0.0;
            for (int s = 0; s < n_samples; s++) {
                sum += P(s, i) * P(s, j);
            }
            G(i, j) = sum;
            G(j, i) = G(i, j);
        }
    }

    // Compute s = P^T y
    Eigen::VectorXd s(M);
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        double sum = 0.0;
        for (int t = 0; t < n_samples; t++) {
            sum += P(t, i) * y(t);
        }
        s(i) = sum;
    }

    // Compute y^T y
    double y2 = 0.0;
    #pragma omp parallel for reduction(+:y2)
    for (int t = 0; t < n_samples; t++) {
        y2 += y(t) * y(t);
    }

    std::vector<TripletCandidate> global_best;

    #pragma omp parallel
    {
        std::vector<TripletCandidate> local_best;
        local_best.reserve(top_n + 1);

        #pragma omp for collapse(2) schedule(dynamic, 16)
        for (int i = 0; i < M - 2; i++) {
            for (int j = i + 1; j < M - 1; j++) {
                for (int k = j + 1; k < M; k++) {
                    Eigen::Matrix3d G3;
                    G3 << G(i, i), G(i, j), G(i, k),
                          G(i, j), G(j, j), G(j, k),
                          G(i, k), G(j, k), G(k, k);
                    Eigen::Vector3d s3(s(i), s(j), s(k));

                    Eigen::Vector3d w = G3.colPivHouseholderQr().solve(s3);
                    if (!std::isfinite(w(0)) || !std::isfinite(w(1)) ||
                        !std::isfinite(w(2))) {
                        continue;
                    }

                    double mse =
                        (y2 - 2.0 * w.dot(s3) + w.dot(G3 * w)) / n_samples;
                    if (mse < 0.0) mse = 0.0;
                    double rmse = std::sqrt(mse);

                    TripletCandidate cand{rmse, i, j, k, w(0), w(1), w(2)};

                    if ((int)local_best.size() < top_n) {
                        local_best.push_back(cand);
                        std::push_heap(local_best.begin(), local_best.end());
                    } else if (rmse < local_best.front().rmse) {
                        std::pop_heap(local_best.begin(), local_best.end());
                        local_best.back() = cand;
                        std::push_heap(local_best.begin(), local_best.end());
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (const auto& cand : local_best) {
                if ((int)global_best.size() < top_n) {
                    global_best.push_back(cand);
                    std::push_heap(global_best.begin(), global_best.end());
                } else if (cand.rmse < global_best.front().rmse) {
                    std::pop_heap(global_best.begin(), global_best.end());
                    global_best.back() = cand;
                    std::push_heap(global_best.begin(), global_best.end());
                }
            }
        }
    }

    std::sort(
        global_best.begin(),
        global_best.end(),
        [](const TripletCandidate& a, const TripletCandidate& b) {
            return a.rmse < b.rmse;
        }
    );

    const int n_kept = static_cast<int>(global_best.size());

    std::vector<std::tuple<int,int,int,double,double,double,double>> metadata;
    metadata.reserve(n_kept);

    py::array_t<double> preds_out({n_samples, n_kept});
    auto preds = preds_out.mutable_unchecked<2>();

    for (int col = 0; col < n_kept; col++) {
        const auto& cand = global_best[col];

        metadata.emplace_back(
            cand.i, cand.j, cand.k,
            cand.wi, cand.wj, cand.wk,
            cand.rmse
        );

        for (int row = 0; row < n_samples; row++) {
            preds(row, col) =
                cand.wi * P(row, cand.i) +
                cand.wj * P(row, cand.j) +
                cand.wk * P(row, cand.k);
        }
    }

    return std::make_tuple(metadata, preds_out);
}


PYBIND11_MODULE(_triplet_search, m) {
    m.doc() =
    "Exact exhaustive top-N OLS-weighted triplet search with OpenMP and Eigen";

    m.def("find_top_triplets", &find_top_triplets,
        py::arg("P_in"),
        py::arg("y_in"),
        py::arg("top_n"),
        py::arg("n_threads") = -1);
}
