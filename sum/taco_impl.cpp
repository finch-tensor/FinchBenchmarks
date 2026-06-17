#include "taco.h"
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

namespace fs = std::filesystem;

using namespace taco;

int main(int argc, char **argv){
    auto params = parse(argc, argv);
    Tensor<double> v1 = read(fs::path(params.input)/"v1.ttx", Format({Sparse}), true);
    Tensor<double> v2 = read(fs::path(params.input)/"v2.ttx", Format({Sparse}), true);
    int dim = v1.getDimension(0);
    Tensor<double> sum("sum", {}, Format());

    IndexVar i;
    sum() += v1(i) * v2(i);

    IndexStmt stmt = sum.getAssignment().concretize();

    // std::cerr << "\tDebug: Parallelizie" << std::endl;

    stmt = stmt.parallelize(
        i,
        ParallelUnit::CPUThread,
        OutputRaceStrategy::ParallelReduction // seems to be best option
    );

    // std::cerr << "\tDebug: Compile" << std::endl;

    sum.compile(stmt);

    // std::cerr << "\tDebug: Execute" << std::endl;

    // Assemble output indices and numerically compute the result
    auto time = benchmark(
      [&sum]() {
        sum.setNeedsAssemble(true);
        sum.setNeedsCompute(true);
      },
      [&sum]() {
        sum.assemble();
        sum.compute();
      }
    );

    // write(fs::path(params.input)/"s.ttx", sum);
    double result = ((double*)sum.getStorage().getValues().getData())[0];

    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;
    measurements["result"] = result;
    std::ofstream measurements_file(fs::path(params.output)/"measurements.json");
    measurements_file << measurements;
    measurements_file.close();
    return 0;
}