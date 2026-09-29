#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <vector>
#include "tbb/info.h"
#include "dcsr_add_cpu.h"
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

typedef Eigen::SparseMatrix<float, Eigen::RowMajor, int32_t> SpMat;

struct DCSR {
    std::vector<int32_t> dim_i_indices;
    std::vector<int32_t> dim_j_offsets;
    std::vector<int32_t> dim_j_indices;
    std::vector<float> values;
};

DCSR csr_to_dcsr(const SpMat &M) {
    DCSR d;
    const int32_t *outer = M.outerIndexPtr();
    d.dim_j_offsets.push_back(0);
    for (int32_t i = 0; i < M.rows(); i++) {
        if (outer[i + 1] > outer[i]) {
            d.dim_i_indices.push_back(i);
            d.dim_j_offsets.push_back(outer[i + 1]);
        }
    }
    d.dim_j_indices.assign(M.innerIndexPtr(), M.innerIndexPtr() + M.nonZeros());
    d.values.assign(M.valuePtr(), M.valuePtr() + M.nonZeros());
    return d;
}

template <typename TensorFormat>
TensorFormat as_nacho_tensor(DCSR &d, const SpMat &M) {
    TensorFormat t = {};
    t.dim_i_size = M.rows();
    t.dim_j_size = M.cols();
    t.dim_i_length = d.dim_i_indices.size();
    t.dim_i_indices = d.dim_i_indices.data();
    t.dim_j_offsets = d.dim_j_offsets.data();
    t.dim_j_length = d.dim_j_indices.size();
    t.dim_j_indices = d.dim_j_indices.data();
    t.values = d.values.data();
    t.nnz = d.values.size();
    return t;
}

void free_result(dcsr_add::Z_tensor_format<dcsr_add::index_t, dcsr_add::value_t> &Z) {
    free(Z.dim_i_indices);
    free(Z.dim_j_offsets);
    free(Z.dim_j_indices);
    free(Z.values);
    Z = {};
}

int main(int argc, char **argv){
    auto params = parse(argc, argv);

    // The kernel picks its own thread count from TBB; report it for the record.
    int n_threads = tbb::info::default_concurrency();
    std::cout << n_threads << std::endl;

    SpMat A, B;
    Eigen::loadMarket(A, params.input + "/A.ttx");
    Eigen::loadMarket(B, params.input + "/B.ttx");
    A.makeCompressed();
    B.makeCompressed();

    DCSR A_dcsr = csr_to_dcsr(A);
    DCSR B_dcsr = csr_to_dcsr(B);
    auto a = as_nacho_tensor<dcsr_add::a_tensor_format<dcsr_add::index_t, dcsr_add::value_t>>(A_dcsr, A);
    auto b = as_nacho_tensor<dcsr_add::b_tensor_format<dcsr_add::index_t, dcsr_add::value_t>>(B_dcsr, B);
    dcsr_add::Z_tensor_format<dcsr_add::index_t, dcsr_add::value_t> Z = {};

    auto time = benchmark(
      [&Z]() {
        free_result(Z);
      },
      [&Z, &a, &b]() {
        Z = dcsr_add::dcsr_add_cpu_i32_f32(a, b);
      }
    );

    std::vector<Eigen::Triplet<float, int32_t>> triplets;
    triplets.reserve(Z.dim_j_length);
    for (int32_t p = 0; p < Z.dim_i_length; p++) {
        int32_t i = Z.dim_i_indices[p];
        for (int32_t q = Z.dim_j_offsets[p]; q < Z.dim_j_offsets[p + 1]; q++) {
            triplets.emplace_back(i, Z.dim_j_indices[q], Z.values[q]);
        }
    }
    SpMat C(A.rows(), A.cols());
    C.setFromTriplets(triplets.begin(), triplets.end());
    free_result(Z);

    Eigen::saveMarket(C, params.input + "/C.ttx");
    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;
    measurements["n_threads"] = n_threads;
    std::ofstream measurements_file(params.output + "/measurements.json");
    measurements_file << measurements;
    measurements_file.close();
    return 0;
}
