#include "taco.h"
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

namespace fs = std::filesystem;
using namespace taco;


// parallel random insert not supported by taco

int main(int argc, char **argv){
    auto params = parse(argc, argv);
    Tensor<double> A = read(fs::path(params.input)/"A.ttx", Format({Dense, Sparse}), true);
    Tensor<double> B = read(fs::path(params.input)/"B.ttx", Format({Dense, Sparse}), true);
    int rows = A.getDimension(0);
    int cols = A.getDimension(1);
    Tensor<double> C("C", {rows, cols}, Format({Sparse, Sparse}));
    IndexVar i, j;
    C(i, j) = A(i, j) + B(i, j);
    IndexStmt stmt = C.getAssignment().concretize();
    stmt = stmt.parallelize(
        i,
        ParallelUnit::CPUThread,
        OutputRaceStrategy::ParallelReduction
    );
    C.compile(stmt);
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