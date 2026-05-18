#include "taco.h"
#include <mkl.h>
// #include <mkl_spblas.h>
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

    std::vector<double> nz_vals;
    std::vector<MKL_INT> nz_indices;
    int64_t dim, nnz;
    {
        std::ifstream f(fs::path(params.input) / "v.ttx");
        std::string line;
        while (std::getline(f, line) && line[0] == '%') {}
        std::istringstream(line) >> dim >> nnz;
        nz_vals.reserve(nnz);
        nz_indices.reserve(nnz);
        int64_t idx;
        double val;
        while (f >> idx >> val) {
            nz_indices.push_back(static_cast<MKL_INT>(idx - 1)); // ttx is 1-indexed
            nz_vals.push_back(val);
        }
    }

    std::vector<double> ones(dim, 1.0);
    double out_sum = 0.0;

    // Assemble output indices and numerically compute the result
    auto time = benchmark(
      []() { 
      },
      [&out_sum, &nz_vals, &nz_indices, &ones, &nnz]() {
        // for (int i = 0; i < 10000; i++) {
        //   out_sum += cblas_ddoti(nnz, nz_vals.data(), nz_indices.data(), ones.data());
        // }
        out_sum = cblas_ddoti(nnz, nz_vals.data(), nz_indices.data(), ones.data());
      }
    );

    Tensor<double> s("s", {}, Format());
    s.insert({}, out_sum);
    s.pack();
    write(fs::path(params.input)/"s.ttx", s);

    json measurements;
    // measurements["time"] = time.first / 10000;
    measurements["time"] = time;
    measurements["memory"] = 0;
    std::ofstream measurements_file(fs::path(params.output)/"measurements.json");
    measurements_file << measurements;
    measurements_file.close();
    return 0;
}