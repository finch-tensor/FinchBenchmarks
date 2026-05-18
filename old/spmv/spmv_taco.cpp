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
    {"schedule", required_argument, 0, 's'},
    {0, 0, 0, 0}
  };

  std::string schedule = "row_major";

  // Parse the options
  int option_index = 0;
  int c;
  optind = 1;
  while ((c = getopt_long(params.argc, params.argv, "hs:", long_options, &option_index)) != -1) {
    switch (c) {
      case 'h':
        std::cout << "Options:" << std::endl;
        std::cout << "  -h, --help      Print this help message" << std::endl;
        std::cout << "  -s, --schedule  Execution schedule, from [row-major, column-major]" << std::endl;
        exit(0);
      case 's':
        schedule = optarg;
        break;
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
  Tensor<double> x = read(fs::path(params.input)/"x.ttx", Format({Dense}), true);
  int m = A.getDimension(0);
  int n = A.getDimension(1);
  Tensor<double> y;
  if (schedule == "row-major")
    y = Tensor<double>("y", {m}, Format({Dense}));
  else if (schedule == "column-major")
    y = Tensor<double>("y", {n}, Format({Dense}));
  else {
    std::cerr << "Invalid schedule" << std::endl;
    exit(1);
  }

  IndexVar i, j;

  if (schedule == "row-major")
    y(i) += A(i, j) * x(j);
  else if (schedule == "column-major")
    y(j) += A(i, j) * x(i);
  else {
    std::cerr << "Invalid schedule" << std::endl;
    exit(1);
  }

  //perform an spmv of the matrix in c++

  y.compile();

  // Assemble output indices and numerically compute the result
  auto time = benchmark(
    [&y]() {
      y.setNeedsAssemble(true);
      y.setNeedsCompute(true);
    },
    [&y]() {
      y.assemble();
      y.compute();
    }
  );

  write(fs::path(params.input)/"y.ttx", y);

  json measurements;
  measurements["time"] = time;
  measurements["memory"] = 0;
  std::ofstream measurements_file(fs::path(params.output)/"measurements.json");
  measurements_file << measurements;
  measurements_file.close();
  return 0;
}
