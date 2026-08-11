#include "Halide.h"
#include <algorithm>
#include <stdio.h>
#include <sstream>
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "../../deps/SparseRooflineBenchmark/src/benchmark.hpp"

using namespace Halide;


cv::Mat loadTTX(const std::string& path) {
    std::ifstream f(path);

    std::string line;
    while (std::getline(f, line) && line[0] == '%') {}

    int rows, cols, nnz;
    std::istringstream(line) >> rows >> cols >> nnz;

    cv::Mat M = cv::Mat::zeros(rows, cols, CV_8U);

    int i, j;
    int val;

    while (f >> i >> j >> val)
        M.at<uint8_t>(i-1,j-1) = static_cast<uint8_t>(val);

    return M;
}


void writeHalide3DTTX(const std::string& path, const Buffer<int32_t>& M) {
    std::ofstream out(path);
    out << "%%MatrixMarket tensor coordinate real general\n%\n";

    int dim0 = M.dim(0).extent();
    int dim1 = M.dim(1).extent();
    int dim2 = M.dim(2).extent();

    size_t nnz = 0;
    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            for (int k = 0; k < dim2; k++) {
                if (M(i, j, k) != 0) nnz++;
            }
        }
    }
    
    out << dim0 << " " << dim1 << " " << dim2 << " " << nnz << "\n";

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            for (int k = 0; k < dim2; k++) {
                int32_t val = M(i, j, k);
                if (val != 0) {
                    out << (i + 1) << " " << (j + 1) << " " << (k + 1) << " " << val << "\n";
                }
            }
        }
    }
}


int main(int argc, char **argv) {
    int n_threads = 1;
    std::vector<std::string> filtered_args;
    filtered_args.reserve(argc);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) {
                n_threads = std::stoi(argv[++i]);
            }
            continue;
        }
        filtered_args.push_back(arg);
    }

    std::vector<char*> cstr_args;
    cstr_args.reserve(filtered_args.size() + 1);
    cstr_args.push_back(argv[0]);
    for (auto &arg : filtered_args) {
        cstr_args.push_back(arg.data());
    }
    cstr_args.push_back(nullptr);

    auto params = parse(static_cast<int>(cstr_args.size()) - 1, cstr_args.data());

    std::string threads_str = std::to_string(n_threads);
    setenv("HL_NUM_THREADS", threads_str.c_str(), 1);

    cv::Mat R = loadTTX(params.input + "/R.ttx");
    cv::Mat G = loadTTX(params.input + "/G.ttx");
    cv::Mat B = loadTTX(params.input + "/B.ttx");

    int width = R.cols;
    int height = R.rows;

    Buffer<uint8_t> input_buf(width, height, 3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            input_buf(x, y, 0) = R.at<uint8_t>(y, x);
            input_buf(x, y, 1) = G.at<uint8_t>(y, x);
            input_buf(x, y, 2) = B.at<uint8_t>(y, x);
        }
    }

    ImageParam input(UInt(8), 3, "input");
    Var bin("bin"), c("c"), temp("temp");
    Func histogram("histogram");

    histogram(bin, c, temp) = 0;

    RDom r(0, input.dim(0).extent(), 0, input.dim(1).extent());
    Expr val = input(r.x, r.y, c);
    histogram(val, c, 0) += 1;


    RVar yo("yo"), yi("yi");
    Var u("u");
    int split_factor = n_threads;
    if (split_factor <= 0) split_factor = 1;
    if (split_factor > height) split_factor = height > 0 ? height : 1;

    try {
        histogram.update(0).split(r.y, yo, yi, Expr(split_factor), TailStrategy::GuardWithIf);

        Func local_hist = histogram.update(0).rfactor({{yo, u}});
        local_hist.compute_root().parallel(u);
        histogram.update(0).serial(yo);
    } catch (const Halide::CompileError &e) {
        std::cerr << "Halide compile error during scheduling: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "Error during scheduling: " << e.what() << std::endl;
        return 1;
    }
    

    // local_hist.update(0).vectorize(local_hist.args()[0], 8);

    input.set(input_buf);
    Buffer<int32_t> output_hist(256, 3, 1);

    auto time = benchmark(
        [&]() {
        },
        [&]() {
            histogram.realize(output_hist);
        }
    );

    writeHalide3DTTX(params.output + "/hist.ttx", output_hist);

    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;

    std::ofstream measurements_file(params.output + "/measurements.json");
    measurements_file << measurements;
    measurements_file.close();

    return 0;
}
