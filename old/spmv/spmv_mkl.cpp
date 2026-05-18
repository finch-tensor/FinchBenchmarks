#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <mkl.h>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

int main(int argc, char **argv) {
    mkl_set_num_threads(1);

    auto params = parse(argc, argv);

    Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_A;
    Eigen::VectorXd eigen_x;

    Eigen::loadMarket(eigen_A, (params.input + "/A.ttx").c_str());
    Eigen::SparseMatrix<double> sparseX;
    Eigen::loadMarket(sparseX, (params.input + "/x.ttx").c_str());
    Eigen::MatrixXd denseX = sparseX;
    eigen_x = denseX;

    // Convert Eigen matrix A to MKL format using Eigen's internal data
    const int* outerIndexPtr = eigen_A.outerIndexPtr();
    const int* innerIndexPtr = eigen_A.innerIndexPtr();
    const double* valuePtr = eigen_A.valuePtr();

    MKL_INT *csr_row_pointer = (MKL_INT *)mkl_malloc((eigen_A.rows() + 1) * sizeof(MKL_INT), 64);
    MKL_INT *csr_columns = (MKL_INT *)mkl_malloc(eigen_A.nonZeros() * sizeof(MKL_INT), 64);
    double *csr_values = (double *)mkl_malloc(eigen_A.nonZeros() * sizeof(double), 64);

    for (int i = 0; i <= eigen_A.rows(); ++i) {
        csr_row_pointer[i] = outerIndexPtr[i];
    }

    for (int i = 0; i < eigen_A.nonZeros(); ++i) {
        csr_columns[i] = innerIndexPtr[i];
        csr_values[i] = valuePtr[i];
    }

    sparse_matrix_t A;
    sparse_status_t status = mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, eigen_A.rows(), eigen_A.cols(),
                                                     csr_row_pointer, csr_row_pointer + 1,
                                                     csr_columns, csr_values);
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Failed to create CSR matrix with MKL. Error code: " << status << "\n";
        return -1;
    }

    // Convert Eigen vector eigen_x to raw pointer
    double *x = (double *)mkl_malloc(eigen_x.size() * sizeof(double), 64);
    for (int i = 0; i < eigen_x.size(); ++i) {
        x[i] = eigen_x[i];
    }
    double *y = (double *)mkl_malloc(sizeof(double) * eigen_A.rows(), 64);

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    //mkl_sparse_set_mv_hint(A, SPARSE_OPERATION_NON_TRANSPOSE, descr, 1000);
    //mkl_sparse_optimize(A);

    auto time = benchmark(
        [&x, &y, &descr, &A]() {},
        [&x, &y, &descr, &A]() {
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, 0.0, y);
        }
    );
    // Convert the result vector y to Eigen format
    Eigen::VectorXd eigen_y(eigen_A.rows());
    for (int i = 0; i < eigen_A.rows(); ++i) {
        eigen_y[i] = y[i];
    }

    // Write the Eigen vector to a file
    Eigen::MatrixXd denseY = eigen_y;
    Eigen::SparseMatrix<double> sparseY = denseY.sparseView();
    Eigen::saveMarket(sparseY, (params.input + "/y.ttx").c_str());

    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;
    std::ofstream measurements_file(params.output + "/measurements.json");
    measurements_file << measurements;
    measurements_file.close();
    return 0;
}
