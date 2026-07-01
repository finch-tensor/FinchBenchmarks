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

    cv::Mat M = cv::Mat::zeros(rows, cols, CV_64F);

    int i, j;
    double val;

    while (f >> i >> j >> val)
        M.at<double>(i-1,j-1) = val;

    return M;
}


void writeHalide2DTTX(const std::string& path, const Buffer<double>& M) {
    std::ofstream out(path);
    out << "%%MatrixMarket matrix coordinate real general\n%\n";

    int width = M.dim(0).extent();
    int height = M.dim(1).extent();

    size_t nnz = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (M(x,y) != 0) nnz++;
        }
    }
    
    out << height << " " << width << " " << nnz << "\n";

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double val = M(x, y);
            if (val != 0) {
                out << (y + 1) << " " << (x + 1) << " " << val << "\n";
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

    cv::Mat A = loadTTX(params.input + "/A.ttx");
    cv::Mat B = loadTTX(params.input + "/B.ttx");

    int width = A.cols;
    int height = A.rows;

    Buffer<double> buf_A(width, height);
    Buffer<double> buf_B(width, height);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            buf_A(x, y) = A.at<double>(y, x);
            buf_B(x, y) = B.at<double>(y, x);
        }
    }

    ImageParam input_A(Float(64), 2, "input_A");
    ImageParam input_B(Float(64), 2, "input_B");
    
    Var x("x"), y("y");
    Func add_result("add_result");
    
    // Define element-wise addition
    add_result(x, y) = cast<double>(input_A(x, y)) + cast<double>(input_B(x, y));

    try {
        add_result.parallel(x);
        // add_result.vectorize(x, 8);
    } catch (const Halide::CompileError &e) {
        std::cerr << "Halide compile error during scheduling: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "Error during scheduling: " << e.what() << std::endl;
        return 1;
    }

    input_A.set(buf_A);
    input_B.set(buf_B);
    Buffer<double> output(width, height);

    auto time = benchmark(
        [&]() {
        },
        [&]() {
            add_result.realize(output);
        }
    );

    writeHalide2DTTX(params.output + "/C.ttx", output);

    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;

    std::ofstream measurements_file(params.output + "/measurements.json");
    measurements_file << measurements;
    measurements_file.close();

    return 0;
}
