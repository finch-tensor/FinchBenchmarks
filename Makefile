CC = gcc
CXX = g++
LD = ld
CXXFLAGS += -std=c++17 -O3 -march=native
LDLIBS +=

ifeq ("$(shell uname)","Darwin")
export NPROC_VAL := $(shell sysctl -n hw.logicalcpu_max )
else
export NPROC_VAL := $(shell lscpu -p | egrep -v '^\#' | wc -l)
endif

SPMV_TACO = spmv/spmv_taco
# SPMV_EIGEN = spmv/spmv_eigen
# SPMV_MKL = spmv/spmv_mkl

SPGEMM_TACO = spgemm/spgemm_taco
# SPGEMM_EIGEN = spgemm/spgemm_eigen
# SPGEMM_MKL = spgemm/spgemm_mkl

# BFS_LAGRAPH = graphs/bfs_lagraph
# BELLMAN_FORD_LAGRAPH = graphs/bellmanford_lagraph

SPARSE_BENCH_DIR = deps/SparseRooflineBenchmark
SPARSE_BENCH_CLONE = $(SPARSE_BENCH_DIR)/.git
SPARSE_BENCH = $(SPARSE_BENCH_DIR)/build/hello

TACO_DIR = deps/taco
TACO_CLONE = $(TACO_DIR)/.git
TACO = $(TACO_DIR)/build/hello
TACO_CXXFLAGS = -I$(TACO_DIR)/include -I$(TACO_DIR)/src
TACO_LDLIBS = -L$(TACO_DIR)/build/lib -ltaco -ldl

# GRAPHBLAS_DIR = deps/GraphBLAS
# GRAPHBLAS_CLONE = $(GRAPHBLAS_DIR)/.git
# GRAPHBLAS = deps/GraphBLAS/build/hello
# GRAPHBLAS_CXXFLAGS = -I$(GRAPHBLAS_DIR)/Include
# GRAPHBLAS_LDLIBS = -L$(GRAPHBLAS_DIR)/build -lgraphblas

# LAGRAPH_DIR = deps/LAGraph
# LAGRAPH_CLONE = $(LAGRAPH_DIR)/.git
# LAGRAPH = deps/LAGraph/build/hello
# LAGRAPH_CXXFLAGS = -I$(LAGRAPH_DIR)/include
# LAGRAPH_LDLIBS = -L$(LAGRAPH_DIR)/build/src -llagraph -L$(LAGRAPH_DIR)/build/experimental -llagraphx

# EIGEN_DIR = deps/eigen
# EIGEN_CLONE = $(EIGEN_DIR)/.git
# EIGEN_CXXFLAGS = -I$(EIGEN_DIR)

# MKLROOT = deps/intel/mkl/2024.2
# MKL_CXXFLAGS = -I$(MKLROOT)/include
# MKL_LDLIBS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential

# CORA_DIR = deps/cora
# CORA_Z3 = $(CORA_DIR)/z3/hello
# CORA_LLVM = $(CORA_DIR)/llvm/hello
# CORA_CLONE = $(CORA_DIR)/.git
# CORA = deps/cora/build/libtvm.so

ALL_TARGETS = $(SPMV_TACO) $(SPGEMM_TACO) 
# $(SPMV_EIGEN) $(SPGEMM_EIGEN) $(GRAPHBLAS) $(LAGRAPH) graphs/rmat_gen $(BFS_LAGRAPH) $(BELLMAN_FORD_LAGRAPH)

# ifeq ($(shell uname -m), x86_64)
# 	ALL_TARGETS += $(SPMV_MKL) $(SPGEMM_MKL) $(CORA)
# endif

all: $(ALL_TARGETS)

clean:
	rm -f $(ALL_TARGETS)
	rm -rf *.o *.dSYM *.trace

$(SPARSE_BENCH_CLONE): 
	git submodule update --init $(SPARSE_BENCH_DIR)

$(SPARSE_BENCH): $(SPARSE_BENCH_CLONE)
	mkdir -p $(SPARSE_BENCH) ;\
	touch $(SPARSE_BENCH)

$(TACO_CLONE): 
	git submodule update --init $(TACO_DIR)

$(TACO): $(TACO_CLONE)
	cd $(TACO_DIR) ;\
	mkdir -p build ;\
	cd build ;\
	cmake -DPYTHON=false -DCMAKE_BUILD_TYPE=Release -DOPENMP=ON .. ;\
	make taco -j$(NPROC_VAL) ;\
	touch hello

# $(GRAPHBLAS_CLONE): 
# 	git submodule update --init $(GRAPHBLAS_DIR)

# $(GRAPHBLAS): $(GRAPHBLAS_CLONE)
# 	cd $(GRAPHBLAS_DIR) ;\
# 	make JOBS=32 ;\
# 	touch build/hello

# $(LAGRAPH_CLONE): 
# 	git submodule update --init $(LAGRAPH_DIR)

# $(LAGRAPH): $(LAGRAPH_CLONE)
# 	cd $(LAGRAPH_DIR) ;\
# 	GRAPHBLAS_ROOT=$(GRAPHBLAS_DIR) make ;\
# 	touch build/hello

# $(EIGEN_CLONE): 
# 	git submodule update --init $(EIGEN_DIR)

# $(CORA_CLONE):
# 	git submodule update --init --recursive $(CORA_DIR)

# $(CORA_LLVM):
# 	cd $(CORA_DIR) ;\
# 	curl -L https://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-pc-linux-gnu.tar.xz -o llvm.tar.xz ;\
# 	tar -xf llvm.tar.xz ;\
# 	mv clang+llvm-9.0.0-x86_64-pc-linux-gnu llvm ;\
# 	touch llvm/hello

# $(CORA_Z3):
# 	cd $(CORA_DIR) ;\
# 	curl -L https://github.com/Z3Prover/z3/releases/download/z3-4.8.8/z3-4.8.8-x64-ubuntu-16.04.zip -o z3.zip ;\
# 	unzip -q z3.zip ;\
# 	mv z3-4.8.8-x64-ubuntu-16.04 z3 ;\
# 	cd z3/bin ;\
# 	ln -s libz3.so libz3.so.4.8 ;\
# 	cd - ;\
# 	touch z3/hello

# $(CORA): $(CORA_CLONE) $(CORA_LLVM) $(CORA_Z3)
# 	cd $(CORA_DIR) ;\
# 	mkdir -p build ;\
# 	cp ../cora_config.cmake build/config.cmake ;\
# 	cd build ;\
#         LLVM_PATH=$(shell pwd)/$(CORA_DIR)/llvm \
#         USE_LLVM=$(shell pwd)/$(CORA_DIR)/llvm/bin/llvm-config \
#         USE_MKL_PATH=$(shell pwd)/$(MKLROOT) \
# 	LD_LIBRARY_PATH=$(shell pwd)/$(CORA_DIR)/z3/bin:$(LD_LIBRARY_PATH) \
# 	CPATH=$(shell pwd)/$(CORA_DIR)/z3/include:$(CPATH) \
# 	CPLUS_INCLUDE_PATH=$(shell pwd)/$(CORA_DIR)/z3/include:$(CPLUS_INCLUDE_PATH) \
# 	Z3_INCLUDE=$(shell pwd)/$(CORA_DIR)/z3/include \
# 	bash -c 'source $(shell pwd)/deps/intel/setvars.sh; cmake -DZ3_LIBRARY=$(shell pwd)/$(CORA_DIR)/z3/bin/libz3.so .. && make -j8 tvm'

spgemm/spgemm_taco: $(SPARSE_BENCH) $(TACO) spgemm/spgemm_taco.cpp
	$(CXX) $(CXXFLAGS) $(TACO_CXXFLAGS) -o $@ spgemm/spgemm_taco.cpp $(LDLIBS) $(TACO_LDLIBS)

# spgemm/spgemm_eigen: $(SPARSE_BENCH) $(EIGEN_CLONE) spgemm/spgemm_eigen.cpp
# 	$(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) -o $@ spgemm/spgemm_eigen.cpp

spmv/spmv_taco: $(SPARSE_BENCH) $(TACO) spmv/spmv_taco.cpp
	$(CXX) $(CXXFLAGS) $(TACO_CXXFLAGS) -o $@ spmv/spmv_taco.cpp $(LDLIBS) $(TACO_LDLIBS)

# spmv/spmv_eigen: $(SPARSE_BENCH) $(EIGEN_CLONE) spmv/spmv_eigen.cpp
# 	$(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) -o $@ spmv/spmv_eigen.cpp

# spmv/spmv_mkl: $(SPARSE_BENCH) spmv/spmv_mkl.cpp
# 	bash -c 'source deps/intel/setvars.sh; $(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) $(MKL_CXXFLAGS) -o $@ spmv/spmv_mkl.cpp $(LDLIBS) $(MKL_LDLIBS)'

# spgemm/spgemm_mkl: $(SPARSE_BENCH) spgemm/spgemm_mkl.cpp
# 	bash -c 'source deps/intel/setvars.sh; $(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) $(MKL_CXXFLAGS) -o $@ spgemm/spgemm_mkl.cpp $(LDLIBS) $(MKL_LDLIBS)'

# graphs/rmat_gen: graphs/rmat_gen.cpp
# 	$(CXX) $(CXXFLAGS) -o graphs/rmat_gen graphs/rmat_gen.cpp

# graphs/bellmanford_lagraph: graphs/bellmanford_lagraph.cpp
# 	$(CXX) $(CXXFLAGS) $(GRAPHBLAS_CXXFLAGS) $(LAGRAPH_CXXFLAGS) -o graphs/bellmanford_lagraph graphs/bellmanford_lagraph.cpp $(GRAPHBLAS_LDLIBS) $(LAGRAPH_LDLIBS)

# graphs/bfs_lagraph: graphs/bfs_lagraph.cpp
# 	$(CXX) $(CXXFLAGS) $(GRAPHBLAS_CXXFLAGS) $(LAGRAPH_CXXFLAGS) -o graphs/bfs_lagraph graphs/bfs_lagraph.cpp $(GRAPHBLAS_LDLIBS) $(LAGRAPH_LDLIBS)
