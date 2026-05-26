#include "taco.h"
#include <mkl.h>
#include <omp.h>
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
    mkl_set_num_threads(1);
    omp_set_num_threads(n_threads);

    // Formats
    Format csf({Sparse,Sparse,Sparse});
    Format rm({Dense,Dense});
    
    Tensor<double> B = read(fs::path(params.input) / "B.ttx", csf, true);
    Tensor<double> C = read(fs::path(params.input)/"C.ttx", rm, true);
    Tensor<double> D = read(fs::path(params.input)/"D.ttx", rm, true);

    int I = B.getDimension(0);
    int K = B.getDimension(1);
    int L = B.getDimension(2);
    int R = C.getDimension(1);

    // B(i,k,l) -> B_(1)(i, k*L+l)

    int cols_matricized = K * L;

    std::vector<int> rowptr(I + 1, 0);

    // Count nnz per row
    for (auto &[coords, val] : iterate<double>(B)) {
        int i = coords[0];
        rowptr[i + 1]++;
    }

    // Prefix sum
    for (int i = 0; i < I; i++) {
        rowptr[i + 1] += rowptr[i];
    }

    int nnz = rowptr[I];

    std::vector<int> colidx(nnz);
    std::vector<double> vals(nnz);
    std::vector<int> offset = rowptr;

    // Fill CSR structure
    for (auto &[coords, val] : iterate<double>(B)) {

        int i = coords[0];
        int k = coords[1];
        int l = coords[2];

        int p = offset[i]++;

        colidx[p] = k * L + l;
        vals[p] = val;
    }

        sparse_matrix_t B_mkl;

    mkl_sparse_d_create_csr(&B_mkl, SPARSE_INDEX_BASE_ZERO, I, cols_matricized, 
      rowptr.data(), rowptr.data() + 1, colidx.data(), vals.data());

    double* C_vals = (double*)C.getStorage().getValues().getData();

    double* D_vals = (double*)D.getStorage().getValues().getData();

    std::vector<double> A(I * R, 0.0);

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    auto time = benchmark(
        [&A]() {
          std::fill(A.begin(), A.end(), 0.0);
        },
        [&A, &rowptr, &colidx, &vals, K, L, R, I, C_vals, D_vals]() {
          #pragma omp parallel
          {
            std::vector<double> tmp(R);

            #pragma omp for
              for (int i = 0; i < I; i++) {
                double* A_row = &A[i * R];

                for (int p = rowptr[i]; p < rowptr[i + 1]; p++) {
                  int col = colidx[p];

                  int k = col / L;
                  int l = col % L;

                  double b_val = vals[p];

                  const double* C_row = &C_vals[k * R];

                  const double* D_row = &D_vals[l * R];

                  // tmp = C_row .* D_row
                  vdMul(R, C_row, D_row, tmp.data());

                  // A_row += b_val * tmp
                  cblas_daxpy(R, b_val, tmp.data(), 1, A_row, 1);
                }
              }
          }
      }
    );

    Tensor<double> A_tensor("A", {I, R}, rm);

    for (int i = 0; i < I; i++) {
      for (int r = 0; r < R; r++) {
        double val = A[i * R + r];

        if (val != 0.0) {
          A_tensor.insert({i, r}, val);
        }
      }
    }

    A_tensor.pack();

    write(fs::path(params.output) / "A.ttx", A_tensor);

    mkl_sparse_destroy(B_mkl);

    json measurements;
    measurements["time"] = time;
    // measurements["memory"] = 0;

    std::ofstream measurements_file(fs::path(params.output) / "measurements.json");

    measurements_file << measurements;
    measurements_file.close();

    return 0;
}
