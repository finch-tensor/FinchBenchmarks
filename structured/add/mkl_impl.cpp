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


    Tensor<double> A = read(fs::path(params.input)/"A.ttx", Format({Dense, Dense}), true);
    Tensor<double> B = read(fs::path(params.input)/"B.ttx", Format({Dense, Dense}), true);

    // std::cerr << "\tDebug: Read inputs" << std::endl;

    int rows = A.getDimension(0);
    int cols = A.getDimension(1);

    int matrix_size = rows * cols;
    std::vector<double> a_dense(matrix_size, 0.0);
    std::vector<double> b_dense(matrix_size, 0.0);
    std::vector<double> c_dense(matrix_size, 0.0);

    for (auto& [coords, val] : iterate<double>(A)) {
        a_dense[coords[0] * cols + coords[1]] = val;
    }
    for (auto& [coords, val] : iterate<double>(B)) {
        b_dense[coords[0] * cols + coords[1]] = val;
    }
    

    auto time = benchmark(
      [&]() {
      },
      [&]() {
        cblas_dcopy(matrix_size, a_dense.data(), 1, c_dense.data(), 1);
        cblas_daxpy(matrix_size, 1.0, b_dense.data(), 1, c_dense.data(), 1);
      }
    );

    Tensor<double> C("C", {rows, cols}, Format({Dense, Dense}));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            C.insert({i, j}, c_dense[i * cols + j]);
        }
    }
    C.pack();
    write(fs::path(params.input)/"C.ttx", C);
    
    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;
    std::ofstream measurements_file(fs::path(params.output)/"measurements.json");
    measurements_file << measurements;
    measurements_file.close();
    return 0;
}