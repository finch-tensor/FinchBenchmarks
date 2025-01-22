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
char msg [LEN] ;

//typedef uint64_t GrB_Index ;


int main(int argc, char **argv)
{
	auto params = parse(argc, argv);

	GrB_Vector distances = NULL, parents = NULL, hops = NULL;
	GrB_Matrix A = NULL, A_orig = NULL ;

	GrB_Info info ;

	//------------------------------------------------------------------------------
	// setup: start a test
	//------------------------------------------------------------------------------

    LAGraph_Init (msg);

	auto filename = params.input + "/A.ttx";
	FILE *f = fopen (filename.c_str(), "r") ;
	LAGraph_MMRead (&A_orig, f, msg) ;
	fclose (f) ;
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
		[&distances, &parents, &hops]() {
			GrB_Vector_free(&distances) ;
			GrB_Vector_free(&parents) ;
			GrB_Vector_free(&hops) ;
		},
		[&distances, &parents, &hops, &A_orig, &s]() {
			LAGraph_BF_full1a (&distances, &parents, &hops, A_orig, s);
		}
	);

	GrB_Matrix D;
	GrB_Matrix_new(&D, GrB_FP64, n, 1);
	for (GrB_Index i = 0; i < n; i++) {
		double val;
		GrB_Vector_extractElement_FP64(&val, distances, i);
		GrB_Matrix_setElement_FP64(D, val, i, 0);
	}
	FILE *distances_file = fopen((params.output + "/distances.mtx").c_str(), "w");
	LAGraph_MMWrite(D, distances_file, NULL, msg);
	fclose(distances_file);
	GrB_Matrix_free(&D);

	GrB_Matrix P;
	GrB_Matrix_new(&P, GrB_INT64, n, 1);
	for (GrB_Index i = 0; i < n; i++) {
		int64_t val;
		GrB_Vector_extractElement_INT64(&val, parents, i);
		GrB_Matrix_setElement_INT64(P, val, i, 0);
	}
	FILE *parents_file = fopen((params.output + "/parents.mtx").c_str(), "w");
	LAGraph_MMWrite(P, parents_file, NULL, msg);
	fclose(parents_file);
	GrB_Matrix_free(&P);


	json measurements;
	measurements["time"] = time;
	measurements["memory"] = 0;
	std::ofstream measurements_file(params.output + "/measurements.json");
	measurements_file << measurements;
	measurements_file.close();

    LAGraph_Finalize (msg);

	return 0;
}