#include "../../deps/SparseRooflineBenchmark/src/benchmark.hpp"
#include "../mlir.hpp"
#include <cstdlib>
#include <iostream>
#include <stdlib.h>

int main() {
  void *A = load_csr_mtx("data/square.mtx");
  void *b = load_dense_vec_mtx("data/dense.mtx");

  // TODO: get from mtx files?
  int64_t m = 4;
  double *xd = (double *)calloc(m, sizeof(double));
  MemRefDescriptor<double, 1> x{xd, xd, 0, {m}, {1}};

  auto time = benchmark([&] { memset(xd, 0, m * sizeof(double)); },
                        [&] { _mlir_ciface_spmv(A, b, &x); });
  std::cout << "time: " << time << std::endl;

  std::cout << "x:" << std::endl;
  for (size_t i = 0; i < m; i++)
    std::cout << *(xd + i) << std::endl;

  free(xd);
  return 0;
}