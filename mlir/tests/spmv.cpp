#include "../../deps/SparseRooflineBenchmark/src/benchmark.hpp"
#include "../mlir.hpp"
#include <stdlib.h>
#include <cstdlib>
#include <iostream>

int main() {
  void *A = load_csr_mtx("data/square.mtx");
  void *b = load_dense_vec_mtx("data/dense.mtx");

  int64_t m = 4;
  double *xd = (double *)calloc(m, sizeof(double));
  MemRefDescriptor<double, 1> x{xd, xd, 0, {m}, {1}};

  auto time = benchmark([] {}, [&] { _mlir_ciface_spmv(A, b, &x); });
  std::cout << time << std::endl;

  free(xd);
  return 0;
}