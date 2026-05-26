#include "taco.h"
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <random>
#include <filesystem>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

namespace fs = std::filesystem;
using namespace taco;

int main(int argc, char **argv){
    auto params = parse(argc, argv);

    // Formats
    Format csf({Sparse,Sparse,Sparse});
    Format rm({Dense,Dense});

    // Load tensor
    Tensor<double> B = read(fs::path(params.input)/"B.ttx", csf, true);
    Tensor<double> C = read(fs::path(params.input)/"C.ttx", rm, true);
    Tensor<double> D = read(fs::path(params.input)/"D.ttx", rm, true);

    // std::cout << C << std::endl;
    // std::cout << D << std::endl;

    int I = B.getDimension(0);
    int K = B.getDimension(1);
    int L = B.getDimension(2);
    int R = C.getDimension(1);

    // Output
    Tensor<double> A("A", {I, R}, rm);

    // Index notation
    IndexVar i, j, k, l;
    A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    // Extract statement
    IndexStmt stmt = A.getAssignment().concretize();

    stmt = stmt.parallelize(
        i,  // parallel over rows
        ParallelUnit::CPUThread,
        OutputRaceStrategy::NoRaces
    );

    // Compile
    A.compile(stmt);

    // Benchmark
    auto time = benchmark(
      [&A]() {
        A.setNeedsAssemble(true);
        A.setNeedsCompute(true);
      },
      [&A]() {
        A.assemble();
        A.compute();
      }
    );

    write(fs::path(params.input)/"A.ttx", A);

    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;

    std::ofstream measurements_file(fs::path(params.output)/"measurements.json");
    measurements_file << measurements;
    measurements_file.close();

    return 0;
}
