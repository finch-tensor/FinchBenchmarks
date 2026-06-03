#include <opencv2/opencv.hpp>
#include <sstream>
#include <omp.h>
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <vector>
#include <fstream>
#include "../../deps/SparseRooflineBenchmark/src/benchmark.hpp"

using namespace cv;


Mat loadTTX(const std::string& path)
{
    std::ifstream f(path);

    std::string line;
    while (std::getline(f, line) && line[0] == '%') {}

    int rows, cols, nnz;
    std::istringstream(line) >> rows >> cols >> nnz;

    Mat M = Mat::zeros(rows, cols, CV_64F);

    int i, j;
    double val;

    while (f >> i >> j >> val)
        M.at<double>(i-1,j-1) = val;

    return M;
}

void writeTTX(const std::string& path, const cv::Mat& M)
{
    std::ofstream out(path);

    int rows = M.rows;
    int cols = M.cols;

    size_t nnz = 0;

    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (M.at<double>(r,c) != 0.0)
                nnz++;

    out << "%%MatrixMarket matrix coordinate real general\n";

    out << rows << " "
        << cols << " "
        << nnz << "\n";

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            double val = M.at<double>(r,c);

            if (val != 0.0)
            {
                out << (r + 1) << " "
                    << (c + 1) << " "
                    << val << "\n";
            }
        }
    }
}


int main(int argc, char **argv)
{
    auto params = parse(argc, argv);

    int n_threads = params.max_threads;
    omp_set_num_threads(n_threads);

    Mat A = loadTTX(params.input + "/A.ttx");
    Mat B = loadTTX(params.input + "/B.ttx");
    Mat C(A.rows, A.cols, CV_64F);

    auto time = benchmark(
        [&]() {
        },
        [&]() {
            add(A, B, C);
        }
    );

    writeTTX(params.output + "/C.ttx", C);

    json measurements;
    measurements["time"] = time.first;
    measurements["memory"] = 0;

    std::ofstream measurements_file(params.output + "/measurements.json");

    measurements_file << measurements;
    measurements_file.close();

    return 0;
}
