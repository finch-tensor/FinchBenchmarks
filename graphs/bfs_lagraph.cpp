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

#define LEN 512
char msg [LEN] ;

int main(int argc, char **argv)
{
	auto params = parse(argc, argv);

	GrB_Vector parents = NULL;
	GrB_Matrix A = NULL;
	LAGraph_Graph G = NULL;

	GrB_Info info ;

	//------------------------------------------------------------------------------
	// setup: start a test
	//------------------------------------------------------------------------------

    LAGraph_Init (msg);

	auto filename = params.input + "/A.ttx";
	FILE *f = fopen (filename.c_str(), "r") ;
	LAGraph_MMRead (&A, f, msg) ;
	fclose (f) ;
	GrB_Index nvals ;
	GrB_Matrix_nvals (&nvals, A) ;
	GrB_Index nrows, ncols ;
	GrB_Matrix_nrows (&nrows, A) ;
	GrB_Matrix_ncols (&ncols, A) ;
	GrB_Index n = nrows ;
	GrB_Index s = 0 ;

	LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, msg);
	LAGraph_Cached_AT(G, msg);
	LAGraph_Cached_OutDegree(G, msg);

	auto time = benchmark(
		[&parents, &n]() {
			GrB_Vector_free(&parents) ;
		},
		[&parents, &G, &s]() {
			LAGr_BreadthFirstSearch (NULL, &parents, G, s, msg);
		}
	);

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