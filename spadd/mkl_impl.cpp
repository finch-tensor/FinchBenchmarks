#include "taco.h"
#include <mkl.h>
#include <mkl_spblas.h>
#include <vector>
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

namespace fs = std::filesystem;
using namespace taco;

int main(int argc, char **argv){
    auto params = parse(argc, argv);

    int n_threads = mkl_get_max_threads();

    for (int i = 0; i < params.argc; i++) {
        std::string arg = params.argv[i];
        if (arg == "-t" || arg == "--threads") {
            if (i + 1 < params.argc) {
                n_threads = std::stoi(params.argv[i + 1]);
            }
        }
    }
    mkl_set_num_threads(n_threads);


    Tensor<double> A = read(fs::path(params.input)/"A.ttx", Format({Dense, Sparse}), true);
    Tensor<double> B = read(fs::path(params.input)/"B.ttx", Format({Dense, Sparse}), true);

    // std::cerr << "\tDebug: Read inputs" << std::endl;

    int rows = A.getDimension(0);
    int cols = A.getDimension(1);


    // Build MKL sparse handles from taco tensors
    std::vector<int>    a_row, a_col, b_row, b_col;
    std::vector<double> a_vals, b_vals;
    sparse_matrix_t A_mkl, B_mkl;

    auto build_csr = [](Tensor<double>& T, int rows, int cols,
                        std::vector<int>& row_start,
                        std::vector<int>& col_idx,
                        std::vector<double>& vals) {
        sparse_matrix_t handle;
        row_start.assign(rows + 1, 0);
        for (auto& [coords, val] : iterate<double>(T))
            row_start[coords[0] + 1]++;
        for (int i = 0; i < rows; ++i) row_start[i + 1] += row_start[i];
        col_idx.resize(row_start[rows]);
        vals.resize(row_start[rows]);
        std::vector<int> pos = row_start;
        for (auto& [coords, val] : iterate<double>(T)) {
            int p = pos[coords[0]]++;
            col_idx[p] = coords[1];
            vals[p] = val;
        }
        mkl_sparse_d_create_csr(&handle, SPARSE_INDEX_BASE_ZERO,
                                rows, cols,
                                row_start.data(), row_start.data() + 1,
                                col_idx.data(), vals.data());
        return handle;
    };

    A_mkl = build_csr(A, rows, cols, a_row, a_col, a_vals);
    B_mkl = build_csr(B, rows, cols, b_row, b_col, b_vals);
    
    // std::cerr << "\tDebug: Built A, B MKL handle" << std::endl;

    sparse_matrix_t C_mkl = nullptr;
    const double alpha = 1.0;
    
    // std::cerr << "\tDebug: Starting benchmark" << std::endl;

    auto time = benchmark(
      [&C_mkl]() {
        if (C_mkl) {
            mkl_sparse_destroy(C_mkl);
            C_mkl = nullptr;
        }
      },
      [&A_mkl, &B_mkl, &C_mkl, &alpha]() {
        mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, A_mkl, alpha, B_mkl, &C_mkl);
      }
    );

    // std::cerr << "\tDebug: Writing results" << std::endl;
    // Export result and write via taco
    sparse_index_base_t indexing;
    int c_rows, c_cols;
    int *c_row_start, *c_row_end, *c_inner;
    double *c_vals;
    mkl_sparse_d_export_csr(C_mkl, &indexing, &c_rows, &c_cols,
                            &c_row_start, &c_row_end, &c_inner, &c_vals);
    Tensor<double> C("C", {rows, cols}, Format({Dense, Sparse}));
    for (int i = 0; i < rows; ++i) {
        for (int p = c_row_start[i]; p < c_row_end[i]; ++p) {
            C.insert({i, c_inner[p]}, c_vals[p]);
        }
    }
    C.pack();
    write(fs::path(params.input)/"C.ttx", C);
    mkl_sparse_destroy(A_mkl);
    mkl_sparse_destroy(B_mkl);
    mkl_sparse_destroy(C_mkl);
    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;
    std::ofstream measurements_file(fs::path(params.output)/"measurements.json");
    measurements_file << measurements;
    measurements_file.close();
    return 0;
}