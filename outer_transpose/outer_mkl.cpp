#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <mkl.h>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

int main(int argc, char **argv) {

	auto params = parse(argc, argv);

	// Define eigen_A and eigen_B matrices
	Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_A, eigen_B;

	// Load or initialize eigen_A and eigen_B as needed
	Eigen::loadMarket(eigen_A, (params.input + "/A.ttx").c_str());
	Eigen::loadMarket(eigen_B, (params.input + "/B.ttx").c_str());
	MKL_INT m = eigen_A.rows();
	MKL_INT n = eigen_B.rows();

	// Convert Eigen matrix A to MKL format using Eigen's internal data
	const int* outerIndexPtr_A = eigen_A.outerIndexPtr();
	const int* innerIndexPtr_A = eigen_A.innerIndexPtr();
	const double* valuePtr_A = eigen_A.valuePtr();

	MKL_INT *csr_row_pointer_A = (MKL_INT *)mkl_malloc((eigen_A.rows() + 1) * sizeof(MKL_INT), 64);
	MKL_INT *csr_columns_A = (MKL_INT *)mkl_malloc(eigen_A.nonZeros() * sizeof(MKL_INT), 64);
	double *csr_values_A = (double *)mkl_malloc(eigen_A.nonZeros() * sizeof(double), 64);

	for (int i = 0; i <= eigen_A.rows(); ++i) {
		csr_row_pointer_A[i] = outerIndexPtr_A[i];
	}

	for (int i = 0; i < eigen_A.nonZeros(); ++i) {
		csr_columns_A[i] = innerIndexPtr_A[i];
		csr_values_A[i] = valuePtr_A[i];
	}

	sparse_matrix_t A;
	sparse_status_t status = mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, eigen_A.rows(), eigen_A.cols(),
													 csr_row_pointer_A, csr_row_pointer_A + 1,
													 csr_columns_A, csr_values_A);
	if (status != SPARSE_STATUS_SUCCESS) {
		std::cerr << "Failed to create CSR matrix with MKL. Error code: " << status << "\n";
		return -1;
	}

	// Convert Eigen matrix B to MKL format using Eigen's internal data
	const int* outerIndexPtr_B = eigen_B.outerIndexPtr();
	const int* innerIndexPtr_B = eigen_B.innerIndexPtr();
	const double* valuePtr_B = eigen_B.valuePtr();

	MKL_INT *csr_row_pointer_B = (MKL_INT *)mkl_malloc((eigen_B.rows() + 1) * sizeof(MKL_INT), 64);
	MKL_INT *csr_columns_B = (MKL_INT *)mkl_malloc(eigen_B.nonZeros() * sizeof(MKL_INT), 64);
	double *csr_values_B = (double *)mkl_malloc(eigen_B.nonZeros() * sizeof(double), 64);

	for (int i = 0; i <= eigen_B.rows(); ++i) {
		csr_row_pointer_B[i] = outerIndexPtr_B[i];
	}

	for (int i = 0; i < eigen_B.nonZeros(); ++i) {
		csr_columns_B[i] = innerIndexPtr_B[i];
		csr_values_B[i] = valuePtr_B[i];
	}

	sparse_matrix_t B;
	status = mkl_sparse_d_create_csr(&B, SPARSE_INDEX_BASE_ZERO, eigen_B.rows(), eigen_B.cols(),
									 csr_row_pointer_B, csr_row_pointer_B + 1,
									 csr_columns_B, csr_values_B);
	if (status != SPARSE_STATUS_SUCCESS) {
		std::cerr << "Failed to create CSR matrix with MKL. Error code: " << status << "\n";
		return -1;
	}

	matrix_descr descrA, descrB, descrC;

	descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
	descrA.diag = SPARSE_DIAG_NON_UNIT;
	descrB.type = SPARSE_MATRIX_TYPE_GENERAL;
	descrB.diag = SPARSE_DIAG_NON_UNIT;
	descrC.type = SPARSE_MATRIX_TYPE_GENERAL;
	descrC.diag = SPARSE_DIAG_NON_UNIT;

	// Every variant computes C = A * B^T as R = spgemm(X, opX, Y, opY), selected by
	// the positional argument (default 0):
	//
	//   variant  X             opX  Y             opY  C
	//   0        A             N    B             T    R
	//   1        transpose(A)  T    B             T    R
	//   2        A             N    transpose(B)  N    R
	//   3        transpose(A)  T    transpose(B)  N    R
	//   4        B             N    A             T    transpose(R)
	//   5        B             N    transpose(A)  N    transpose(R)
	//   6        transpose(B)  T    A             T    transpose(R)
	//   7        transpose(B)  T    transpose(A)  N    transpose(R)
	//
	// bit 0: X is physically transposed, bit 1: Y is physically transposed,
	// bit 2: operands swapped, so R = C^T. Physical transposes of the inputs and
	// of R are part of the timed region.
	int variant = params.argc > 1 ? std::atoi(params.argv[1]) : 0;
	if (variant < 0 || variant > 7) {
		std::cerr << "Invalid variant " << variant << ", expected 0-7\n";
		return -1;
	}
	const bool transpose_X = variant & 1;
	const bool transpose_Y = variant & 2;
	const bool swap = variant & 4;
	sparse_matrix_t X_src = swap ? B : A;
	sparse_matrix_t Y_src = swap ? A : B;
	matrix_descr descrX = swap ? descrB : descrA;
	matrix_descr descrY = swap ? descrA : descrB;
	sparse_operation_t opX = transpose_X ? SPARSE_OPERATION_TRANSPOSE : SPARSE_OPERATION_NON_TRANSPOSE;
	sparse_operation_t opY = transpose_Y ? SPARSE_OPERATION_NON_TRANSPOSE : SPARSE_OPERATION_TRANSPOSE;

	sparse_matrix_t X = NULL, Y = NULL, R = NULL, C = NULL;
	auto time = benchmark(
		[&]() {
			// Release the previous trial's matrices outside the timed region.
			if (transpose_X && X) mkl_sparse_destroy(X);
			if (transpose_Y && Y) mkl_sparse_destroy(Y);
			if (swap && R) mkl_sparse_destroy(R);
			if (C) mkl_sparse_destroy(C);
			X = Y = R = C = NULL;
			mkl_free_buffers();
		},
		[&]() {
			// convert_csr fails (NOT_SUPPORTED) if the output handle already holds the source.
			if (transpose_X) mkl_sparse_convert_csr(X_src, SPARSE_OPERATION_TRANSPOSE, &X);
			else X = X_src;
			if (transpose_Y) mkl_sparse_convert_csr(Y_src, SPARSE_OPERATION_TRANSPOSE, &Y);
			else Y = Y_src;
			mkl_sparse_sp2m(opX, descrX, X, opY, descrY, Y, SPARSE_STAGE_FULL_MULT, &R);
			if (swap) {
				mkl_sparse_convert_csr(R, SPARSE_OPERATION_TRANSPOSE, &C);
			} else {
				C = R;
			}
			mkl_sparse_order(C);
		}
	);

	MKL_INT *rows_start_C;
	MKL_INT *rows_end_C;
	MKL_INT *columns_C;
	double *values_C;
	sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;

	mkl_sparse_d_export_csr(C, &indexing, &m, &n, &rows_start_C, &rows_end_C, &columns_C, &values_C);

	// Convert MKL matrix C to Eigen format
	Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_C(m, n);
	eigen_C.resizeNonZeros(rows_start_C[m]);

	for (int i = 0; i < m; ++i) {
		eigen_C.outerIndexPtr()[i] = rows_start_C[i];
	}
	eigen_C.outerIndexPtr()[m] = rows_start_C[m];

	for (int i = 0; i < rows_start_C[m]; ++i) {
		eigen_C.innerIndexPtr()[i] = columns_C[i];
		eigen_C.valuePtr()[i] = values_C[i];
	}

	// Save the Eigen matrix to MatrixMarket format
	Eigen::saveMarket(eigen_C, (params.output + "/C.ttx").c_str());

	json measurements;
	measurements["time"] = time;
	measurements["memory"] = 0;
	std::ofstream measurements_file(params.output + "/measurements.json");
	measurements_file << measurements;
	measurements_file.close();

	return 0;
}
