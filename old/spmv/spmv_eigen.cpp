#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

int main(int argc, char **argv) {
	auto params = parse(argc, argv);

	Eigen::SparseMatrix<double> A;
	Eigen::VectorXd x;
	Eigen::VectorXd y;

	Eigen::loadMarket(A, (params.input + "/A.ttx").c_str());
	Eigen::SparseMatrix<double> sparseX;
	Eigen::loadMarket(sparseX, (params.input + "/x.ttx").c_str());
	Eigen::MatrixXd denseX = sparseX;
	x = denseX;


	// Assemble output indices and numerically compute the result
	auto time = benchmark(
		[&A, &x, &y]() { },
		[&A, &x, &y]() {
			y = A * x;
		}
	);

	Eigen::MatrixXd denseY = y;
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
