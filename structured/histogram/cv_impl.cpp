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

    Mat M = Mat::zeros(rows, cols, CV_8U);

    int i, j;
    int val;

    while (f >> i >> j >> val)
        M.at<uint8_t>(i-1,j-1) = static_cast<uint8_t>(val);

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
            if (M.at<int>(r,c) != 0)
                nnz++;

    out << "%%MatrixMarket matrix coordinate real general\n";

    out << rows << " "
        << cols << " "
        << nnz << "\n";

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            int val = M.at<int>(r,c);

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

    int n_threads = omp_get_max_threads();

    for (int i = 0; i < params.argc; i++) {
        std::string arg = params.argv[i];
        if (arg == "-t" || arg == "--threads") {
            if (i + 1 < params.argc) {
                n_threads = std::stoi(params.argv[i + 1]);
            }
        }
    }
    omp_set_num_threads(n_threads);

    Mat A = loadTTX(params.input + "/A.ttx");
    Mat hist(256, 1, CV_32S);

    auto time = benchmark(
        [&]() {
        },
        [&]() {
            int channels[] = {0};
            int histSize[] = {256};
            float range[] = {0,256};
            const float* ranges[] = {range};

            calcHist(&A, 1, channels, Mat(), hist, 1, histSize, ranges);
        }
    );

    writeTTX(params.output + "/hist.ttx", hist);

    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;

    std::ofstream measurements_file(params.output + "/measurements.json");

    measurements_file << measurements;
    measurements_file.close();

    return 0;
}
