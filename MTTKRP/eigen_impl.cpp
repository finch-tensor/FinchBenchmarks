#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <omp.h>
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

namespace fs = std::filesystem;

struct Entry {
    int i;
    int k;
    int l;
    double val;
};

int main(int argc, char **argv){
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
    Eigen::setNbThreads(n_threads);

    std::vector<Entry> B_entries;

    int I, K, L;
    int64_t nnz;

    {
      std::ifstream f(params.input + "/B.ttx");
      std::string line;

      while (std::getline(f, line) && line[0] == '%') {}

      std::istringstream(line) >> I >> K >> L >> nnz;

      B_entries.reserve(nnz);

      int i, k, l;
      double val;

      while (f >> i >> k >> l >> val) {
        B_entries.push_back({i - 1, k - 1, l - 1, val});
      }
    }

    Eigen::MatrixXd C;
    Eigen::MatrixXd D;

    {
      std::ifstream f(params.input + "/C.ttx");
      std::string line;

      while (std::getline(f, line) && line[0] == '%') {}

      int rows, cols, nnz;
      std::istringstream(line) >> rows >> cols >> nnz;

      C = Eigen::MatrixXd::Zero(rows, cols);

      int i, j;
      double val;

      while (f >> i >> j >> val) {
        C(i - 1, j - 1) = val;
      }
    }

    {
        std::ifstream f(params.input + "/D.ttx");
        std::string line;

        while (std::getline(f, line) && line[0] == '%') {}

        int rows, cols, nnz;

        std::istringstream(line) >> rows >> cols >> nnz;

        D = Eigen::MatrixXd::Zero(rows, cols);

        int i, j;
        double val;

        while (f >> i >> j >> val) {
          D(i - 1, j - 1) = val;
        }
    }

    int R = C.cols();

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(I, R);

    auto time = benchmark(
      [&]() {
        A.setZero();
      },
      [&]() {
        #pragma omp parallel
        {
          Eigen::MatrixXd A_local = Eigen::MatrixXd::Zero(I, R);

          #pragma omp for
            for (size_t p = 0; p < B_entries.size(); p++) {
              const auto &e = B_entries[p];

              A_local.row(e.i).array() += e.val * C.row(e.k).array() * D.row(e.l).array();
            }

          #pragma omp critical
            {
              A += A_local;
            }
        }
      }
    );

    {
      std::ofstream fs(params.output + "/A.ttx");

      fs << "%%MatrixMarket matrix coordinate real general\n";

      fs << I << " " << R << " " << (I * R) << "\n";

      for (int i = 0; i < I; i++) {
        for (int r = 0; r < R; r++) {
          fs << i + 1 << " " << r + 1 << " " << std::scientific << A(i,r) << "\n";
        }
      }
    }

    json measurements;

    measurements["time"] = time;
    measurements["memory"] = 0;

    std::ofstream measurements_file(params.output + "/measurements.json");

    measurements_file << measurements;
    measurements_file.close();

    return 0;
}
