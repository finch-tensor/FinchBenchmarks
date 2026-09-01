
#include <string>
#include <fstream>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

extern "C" {
  void *_mlir_ciface_read_matrix(char *filename);
  void *_mlir_ciface_spmv(void *A, void *x);
  int64_t _mlir_ciface_num_rows(void *a);
}

int main(int argc, char **argv) {
    auto params = parse(argc, argv);

    std::string a_path = params.input + "/A.mtx";

    void *A = _mlir_ciface_read_matrix(a_path.data());
    void *C = nullptr;
    int64_t m = _mlir_ciface_num_rows(A);
    std::vector<double> x(m, 0.0);

    auto time = benchmark(
      [&]() {},
      // FIXME: cannot pass raw pointers as MLIR tensor
      [&]() { C = _mlir_ciface_spmv(A, x.data()); }
    );

    // TODO: write x out and verify spmv

    json measurements;
    measurements["time"] = time;

    // TODO: how to measure mem?
    measurements["memory"] = 0;
    std::ofstream f(params.output + "/measurements.json");
    f << measurements;
    return 0;
}
