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

SPGEMM_EIGEN = spgemm/spgemm_eigen
SPGEMM_MKL = spgemm/spgemm_mkl

SPARSE_BENCH_DIR = deps/SparseRooflineBenchmark
SPARSE_BENCH_CLONE = $(SPARSE_BENCH_DIR)/.git
SPARSE_BENCH = $(SPARSE_BENCH_DIR)/build/hello

TACO_DIR = deps/taco
TACO_CLONE = $(TACO_DIR)/.git
TACO = $(TACO_DIR)/build/hello
TACO_CXXFLAGS = -fopenmp -I$(TACO_DIR)/include -I$(TACO_DIR)/src
TACO_LDLIBS = -L$(TACO_DIR)/build/lib -ltaco -ldl

EIGEN_DIR = deps/eigen
EIGEN_CLONE = $(EIGEN_DIR)/.git
EIGEN_CXXFLAGS = -I$(EIGEN_DIR) -fopenmp

MKLROOT = /opt/intel/oneapi/mkl/2025.3
MKL_CXXFLAGS = -I$(MKLROOT)/include
MKL_LDLIBS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5

ALL_TARGETS = $(SPGEMM_EIGEN) spadd/spadd_eigen hadamard/hadamard_eigen mttkrp/mttkrp_taco spmspv/spmspv_eigen

ifeq ($(shell uname -m), x86_64)
	ALL_TARGETS += $(SPGEMM_MKL) spadd/spadd_mkl
endif

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
	cmake -DOPENMP=ON -DPYTHON=false -DCMAKE_BUILD_TYPE=Release .. ;\
	make taco -j$(NPROC_VAL) ;\
	touch hello

$(EIGEN_CLONE):
	git submodule update --init $(EIGEN_DIR)

spgemm/spgemm_eigen: $(SPARSE_BENCH) $(EIGEN_CLONE) spgemm/spgemm_eigen.cpp
	$(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) -o $@ spgemm/spgemm_eigen.cpp

spgemm/spgemm_mkl: $(SPARSE_BENCH) spgemm/spgemm_mkl.cpp
	bash -c 'source /opt/intel/oneapi/setvars.sh; $(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) $(MKL_CXXFLAGS) -o $@ spgemm/spgemm_mkl.cpp $(LDLIBS) $(MKL_LDLIBS)'

spadd/spadd_mkl: spadd/mkl_impl.cpp
	bash -c 'source /opt/intel/oneapi/setvars.sh; $(CXX) $(CXXFLAGS) $(TACO_CXXFLAGS) $(EIGEN_CXXFLAGS) $(MKL_CXXFLAGS) -o $@ spadd/mkl_impl.cpp $(LDLIBS) $(MKL_LDLIBS) $(TACO_LDLIBS)'

spadd/spadd_eigen: $(SPARSE_BENCH) $(EIGEN_CLONE) spadd/eigen_impl.cpp
	$(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) -o $@ spadd/eigen_impl.cpp

hadamard/hadamard_eigen: $(SPARSE_BENCH) $(EIGEN_CLONE) hadamard/eigen_impl.cpp
	$(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) -o $@ hadamard/eigen_impl.cpp

mttkrp/mttkrp_taco: $(SPARSE_BENCH) $(TACO) mttkrp/taco_impl.cpp
	$(CXX) $(CXXFLAGS) $(TACO_CXXFLAGS) -o $@ mttkrp/taco_impl.cpp $(LDLIBS) $(TACO_LDLIBS)

spmspv/spmspv_eigen: $(SPARSE_BENCH) $(EIGEN_CLONE) spmspv/spmspv_eigen.cpp
	$(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) -o $@ spmspv/spmspv_eigen.cpp

