#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include "../../../deps/SparseRooflineBenchmark/src/benchmark.hpp"

#ifdef __cplusplus
extern "C" {
#endif
#include <LAGraph.h>
#include <LAGraphX.h>
#include <GraphBLAS.h>
#ifdef __cplusplus
}
#endif
//#include <LAGraph_test.h>
//#include <LG_Xtest.h>

#define LEN 512
char filename [LEN] ;
char msg [LEN] ;

//typedef uint64_t GrB_Index ;


int main(int argc, char **argv)
{
	auto params = parse(argc, argv);

	uint64_t *II = NULL, *J = NULL ; // for col/row indices of entries in A
	GrB_Vector d5a = NULL, pi5a = NULL, h5a = NULL;
	GrB_Matrix A = NULL, AT = NULL, A_orig = NULL ;
	//GrB_Index *I = NULL, *J = NULL ; // for col/row indices of entries in A
	double *W = NULL, *d = NULL ;
	int64_t *pi = NULL, *pi10 = NULL ;
	int32_t *W_int32 = NULL, *d10 = NULL ;

	GrB_Info info ;

	char *aname = getenv("MATRIX_INPUT") ;
	//if (strlen (aname) == 0) break;
	////TEST_CASE (aname) ;
	snprintf (filename, LEN, "%s", aname) ;
	FILE *f = fopen (filename, "r") ;
	////TEST_CHECK (f != NULL) ;
	LAGraph_MMRead (&A_orig, f, msg) ;
	fclose (f) ;
	////TEST_MSG ("Loading of valued matrix failed") ;
	printf ("\nMatrix: %s\n", aname) ;
	LAGraph_Matrix_Print (A_orig, LAGraph_SHORT, stdout, NULL) ;

	//bool has_negative_cycle  = files [k].has_negative_cycle ;
	//bool has_integer_weights = files [k].has_integer_weights ;
	//int ktrials = (has_negative_cycle) ? 2 : 1 ;

	//----------------------------------------------------------------------
	// get the size of the problem
	//----------------------------------------------------------------------

	GrB_Index nvals ;
	GrB_Matrix_nvals (&nvals, A_orig) ;
	GrB_Index nrows, ncols ;
	GrB_Matrix_nrows (&nrows, A_orig) ;
	GrB_Matrix_ncols (&ncols, A_orig) ;
	GrB_Index n = nrows ;
	LAGraph_Malloc ((void **) &II, nvals, sizeof (GrB_Index), msg) ;
	LAGraph_Malloc ((void **) &J, nvals, sizeof (GrB_Index), msg) ;
	LAGraph_Malloc ((void **) &W, nvals, sizeof (double), msg) ;
	LAGraph_Malloc ((void **) &W_int32, nvals, sizeof (int32_t), msg) ;

	GrB_Matrix_extractTuples_FP64 (II, J, W, &nvals, A_orig) ;
	/* TODO: may need to remove
	if (has_integer_weights)
	{
		OK (GrB_Matrix_extractTuples_INT32 (I, J, W_int32, &nvals,
					A_orig)) ;
	}*/

	//----------------------------------------------------------------------
	// copy the matrix and set its diagonal to 0
	//----------------------------------------------------------------------

	GrB_Matrix_dup (&A, A_orig) ;
	for (GrB_Index i = 0; i < n; i++)
	{
		GrB_Matrix_setElement_FP64 (A, 0, i, i) ;
	}
	GrB_Index s = 0 ;

	auto time = benchmark(
			[&d5a, &pi5a, &h5a]() {
				GrB_Vector_free(&d5a) ;
				GrB_Vector_free(&pi5a) ;
				GrB_Vector_free(&h5a) ;
			},
			[&d5a, &pi5a, &h5a, A_orig, s]() {
				LAGraph_BF_full1a (&d5a, &pi5a, &h5a, A_orig, s);
			}
			);

	/*
	GrB_Index *indices;
	int32_t *values, *arr;
	indices = (GrB_Index *) malloc(n * sizeof(GrB_Index));
	values = (int32_t *) malloc(n * sizeof(int32_t));
	arr = (int32_t *) malloc(n * sizeof(int32_t));
	GrB_Vector_extractTuples_UDT(indices, values, &n, d5a);


	for (GrB_Index i = 0; i < n; i++) {
		arr[i] = 0;
	}

	// Store values in the correct position
	for (GrB_Index i = 0; i < n; i++) {
		arr[indices[i]] = values[i];
	}


	Eigen::VectorXd eigen_y(n);
	for (int i = 0; i < n; ++i) {
		eigen_y[i] = arr[i];
	}

    Eigen::MatrixXd denseY = eigen_y;
    Eigen::SparseMatrix<double> sparseY = denseY.sparseView();
    Eigen::saveMarket(sparseY, (params.input + "/y.ttx").c_str());
	*/



	json measurements;
	measurements["time"] = time;
	measurements["memory"] = 0;
	std::ofstream measurements_file(params.output + "/measurements.json");
	measurements_file << measurements;
	measurements_file.close();

	return 0;

}