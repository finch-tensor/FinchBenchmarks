#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

int main(int argc, char **argv) {
  auto params = parse(argc, argv);

  FILE *fpA = fopen((params.input+"/A.ttx").c_str(), "r");
  FILE *fpB = fopen((params.input+"/B.ttx").c_str(), "r");
  
  Eigen::SparseMatrix<double> A;
	Eigen::loadMarket(A, (params.input + "/A.ttx").c_str());
  Eigen::SparseMatrix<double> B;
	Eigen::loadMarket(B, (params.input + "/B.ttx").c_str());
  Eigen::SparseMatrix<double> C;

  // Assemble output indices and numerically compute the result
  auto time = benchmark(
    [&A, &B, &C]() {
	C = A * B;
    },
    [&A, &B, &C]() {
	C = A * B;
    }
  );

  Eigen::saveMarket(C, (params.output + "/C.ttx").c_str());

  json measurements;
  measurements["time"] = time;
  measurements["memory"] = 0;
  std::ofstream measurements_file(params.output+"/measurements.json");
  measurements_file << measurements;
  measurements_file.close();
  return 0;
}
