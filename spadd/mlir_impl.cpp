#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"
#include "../mlir/mlir.hpp"
#include <fstream>

extern "C" void *_mlir_ciface_spadd(void *A, void *B);
extern "C" void _mlir_ciface_spdelete(void *C);

int main(int argc, char **argv) {
  auto params = parse(argc, argv);

  void *A = load_csr_mtx((params.input + "/A.mtx").c_str());
  void *B = load_csr_mtx((params.input + "/B.mtx").c_str());

  void *C = nullptr;
  auto time = benchmark(
      [&] {
        if (C) {
          _mlir_ciface_spdelete(C);
          C = nullptr;
        }
      },
      [&] { C = _mlir_ciface_spadd(A, B); });

  output_csr(C, (params.input + "/C.tns").c_str());

  json measurements;
  measurements["time"] = time;
  measurements["memory"] = 0;
  std::ofstream measurements_file(params.output + "/measurements.json");
  measurements_file << measurements;
  measurements_file.close();

  return 0;
}
