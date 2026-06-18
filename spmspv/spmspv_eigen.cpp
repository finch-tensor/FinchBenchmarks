#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"
#include <fstream>
#include <sstream>

int main(int argc, char **argv) {
  auto params = parse(argc, argv);

  FILE *fpA = fopen((params.input+"/A.ttx").c_str(), "r");
  FILE *fpB = fopen((params.input+"/v.ttx").c_str(), "r");
  
  Eigen::SparseMatrix<double> A;
	Eigen::loadMarket(A, (params.input + "/A.ttx").c_str());
  Eigen::SparseVector<double> v;
    {
      std::ifstream f(params.input + "/v.ttx");
      std::string line;
      // Skip comment lines
      while (std::getline(f, line) && line[0] == '%') {}
      // First non-comment line: "size nnz"
      int64_t size, nnz;
      std::istringstream(line) >> size >> nnz;
      v.resize(size);
      v.reserve(nnz);
      int64_t idx;
      double val;
      while (f >> idx >> val) {
        v.insert(idx - 1) = val;  // ttx is 1-indexed
      }
    }
  Eigen::SparseVector<double> y;

  // Assemble output indices and numerically compute the result
  auto time = benchmark(
    [&A, &v, &y]() {
	y = A * v;
    },
    [&A, &v, &y]() {
	y = A * v;
    }
  );

  // Write y manually in Matrix Market format
  std::ofstream out(params.output + "/C.ttx");
out << "%%MatrixMarket matrix coordinate real general\n";  // or match whatever fwrite emits for SparseList
out << y.size() << " " << y.nonZeros() << "\n";
for (Eigen::SparseVector<double>::InnerIterator it(y); it; ++it) {
    out << (it.index() + 1) << " " << it.value() << "\n";  // 1-indexed
}

out.close();
  json measurements;
  measurements["time"] = time;
  measurements["memory"] = 0;
  std::ofstream measurements_file(params.output+"/measurements.json");
  measurements_file << measurements;
  measurements_file.close();
  return 0;
}
