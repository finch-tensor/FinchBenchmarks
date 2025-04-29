#include "taco.h"
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

namespace fs = std::filesystem;

using namespace taco;
extern int optind;

int main(int argc, char **argv){
  auto params = parse(argc, argv);

  static struct option long_options[] = {
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}
  };

  // Parse the options
  int option_index = 0;
  int c;
  optind = 1;
  while ((c = getopt_long(params.argc, params.argv, "hs:", long_options, &option_index)) != -1) {
    switch (c) {
      case 'h':
        std::cout << "Options:" << std::endl;
        std::cout << "  -h, --help      Print this help message" << std::endl;
        exit(0);
      case '?':
        // getopt_long already printed an error message
        break;
      default:
        abort();
    }
  }

  // Check that all required options are present
  if (params.input.empty() || params.output.empty()) {
    std::cerr << "Missing required option" << std::endl;
    exit(1);
  }

  Tensor<double> A = read(fs::path(params.input)/"A.ttx", Format({Dense, Sparse}), true);
  Tensor<double> B = read(fs::path(params.input)/"B.ttx", Format({Dense, Sparse}), true);
  int m = A.getDimension(0);
  int n = A.getDimension(1);
  Tensor<double> C = Tensor<double>("C", {m, n}, Format({Dense, Dense}));

  IndexVar i, j;
  C(i, j) = A(i, j) + B(i, j);

  //perform an spadd of the matrix in c++

  C.compile();

  // Assemble output indices and numerically compute the result
  auto time = benchmark(
    [&C]() {
      C.setNeedsAssemble(true);
      C.setNeedsCompute(true);
    },
    [&C]() {
      C.assemble();
      C.compute();
    }
  );

  write(fs::path(params.input)/"C.ttx", C);

  json measurements;
  measurements["time"] = time;
  measurements["memory"] = 0;
  std::ofstream measurements_file(fs::path(params.output)/"measurements.json");
  measurements_file << measurements;
  measurements_file.close();
  return 0;
}
