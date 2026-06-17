#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <omp.h>
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

int main(int argc, char **argv){
    auto params = parse(argc, argv);

    int n_threads = omp_get_max_threads();

    for (int i = 0; i < params.argc; i++) {
        std::string arg = params.argv[i];
        if (arg == "-t" || arg == "--threads") {
            if (i + 1 < params.argc) {
                n_threads = std::stoi(params.argv[i + 1]);
            }
        }
    }
    omp_set_num_threads(n_threads);

    Eigen::SparseVector<double> v1;
    {
      std::ifstream f(params.input + "/v1.ttx");
      std::string line;
      // Skip comment lines
      while (std::getline(f, line) && line[0] == '%') {}
      // First non-comment line: "size nnz"
      int64_t size, nnz;
      std::istringstream(line) >> size >> nnz;
      v1.resize(size);
      v1.reserve(nnz);
      int64_t idx;
      double val;
      while (f >> idx >> val) {
        v1.insert(idx - 1) = val;  // ttx is 1-indexed
      }
    }

    Eigen::SparseVector<double> v2;
    {
      std::ifstream f(params.input + "/v2.ttx");
      std::string line;
      // Skip comment lines
      while (std::getline(f, line) && line[0] == '%') {}
      // First non-comment line: "size nnz"
      int64_t size, nnz;
      std::istringstream(line) >> size >> nnz;
      v2.resize(size);
      v2.reserve(nnz);
      int64_t idx;
      double val;
      while (f >> idx >> val) {
        v2.insert(idx - 1) = val;  // ttx is 1-indexed
      }
    }

    double out_sum = 0.0;

    // Assemble output indices and numerically compute the result
    auto time = benchmark(
      []() {
      },
      [&v1, &v2, &out_sum]() {
        double s = 0.0;
        s = v1.dot(v2);
        out_sum = s;
      }
    );

    {
      std::ofstream fs(params.output + "/s.ttx");
      fs << "%%MatrixMarket matrix coordinate real general\n";
      fs << "1 1 1\n";
      fs << "1 1 " << std::scientific << out_sum << "\n";
    }

    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;
    std::ofstream measurements_file(params.output + "/measurements.json");
    measurements_file << measurements;
    measurements_file.close();
    return 0;
}