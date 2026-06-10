#include <opencv2/opencv.hpp>
#include <sstream>
#include <omp.h>
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <stdexcept>
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

    if (M.dims != 2 && M.dims != 3) {
        throw std::runtime_error("writeTTX only supports 2D or 3D cv::Mat");
    }

    bool isTensor = (M.dims == 3);
    out << (isTensor ? "%%MatrixMarket tensor coordinate real general\n"
                     : "%%MatrixMarket matrix coordinate real general\n");
    out << "%\n";

    size_t nnz = 0;
    auto getValue2D = [&](int r, int c) -> double {
        switch (M.type()) {
            case CV_8U:  return M.at<uint8_t>(r, c);
            case CV_32S: return M.at<int>(r, c);
            case CV_32F: return M.at<float>(r, c);
            case CV_64F: return M.at<double>(r, c);
            default:     return 0.0;
        }
    };
    auto getValue3D = [&](int d0, int d1, int d2) -> double {
        switch (M.type()) {
            case CV_8U:  return M.ptr<uint8_t>(d0, d1)[d2];
            case CV_32S: return M.ptr<int>(d0, d1)[d2];
            case CV_32F: return M.ptr<float>(d0, d1)[d2];
            case CV_64F: return M.ptr<double>(d0, d1)[d2];
            default:     return 0.0;
        }
    };

    if (!isTensor) {
        out << M.rows << " " << M.cols << " ";
        for (int r = 0; r < M.rows; r++) {
            for (int c = 0; c < M.cols; c++) {
                if (getValue2D(r, c) != 0.0) {
                    nnz++;
                }
            }
        }
        out << nnz << "\n";

        for (int r = 0; r < M.rows; r++) {
            for (int c = 0; c < M.cols; c++) {
                double val = getValue2D(r, c);
                if (val != 0.0) {
                    out << (r + 1) << " "
                        << (c + 1) << " "
                        << val << "\n";
                }
            }
        }
    } else {
        out << M.size[0] << " " << M.size[1] << " " << M.size[2] << " ";
        for (int i = 0; i < M.size[0]; i++) {
            for (int j = 0; j < M.size[1]; j++) {
                for (int k = 0; k < M.size[2]; k++) {
                    if (getValue3D(i, j, k) != 0.0) {
                        nnz++;
                    }
                }
            }
        }
        out << nnz << "\n";

        for (int i = 0; i < M.size[0]; i++) {
            for (int j = 0; j < M.size[1]; j++) {
                for (int k = 0; k < M.size[2]; k++) {
                    double val = getValue3D(i, j, k);
                    if (val != 0.0) {
                        out << (i + 1) << " "
                            << (j + 1) << " "
                            << (k + 1) << " "
                            << val << "\n";
                    }
                }
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
    // omp_set_num_threads(n_threads);
    setNumThreads(n_threads); 

    Mat R = loadTTX(params.input + "/R.ttx");
    Mat G = loadTTX(params.input + "/G.ttx");
    Mat B = loadTTX(params.input + "/B.ttx");

    std::vector<Mat> rgb = {B, G, R};   // OpenCV convention is BGR
    Mat image;

    merge(rgb, image);

    int channels[] = {0, 1, 2};

    int histSize[] = {256, 256, 256};

    float range[] = {0,256};

    const float* ranges[] = {range, range, range};
    
    Mat hist;

    auto time = benchmark(
        [&]() {
        },
        [&]() {
            calcHist(&image, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);
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
