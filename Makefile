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
SPMV_EIGEN = spmv/spmv_eigen
SPMV_MKL = spmv/spmv_mkl

SPGEMM_TACO = spgemm/spgemm_taco
SPGEMM_EIGEN = spgemm/spgemm_eigen
SPGEMM_MKL = spgemm/spgemm_mkl

ALL_TARGETS = $(SPMV_TACO) $(SPGEMM_TACO) $(SPMV_EIGEN) $(SPGEMM_EIGEN)

ifeq ($(shell uname -m), x86_64)
	ALL_TARGETS += $(SPMV_MKL) $(SPGEMM_MKL) $(SPMV_CORA)
endif

all: $(ALL_TARGETS)

SPARSE_BENCH_DIR = deps/SparseRooflineBenchmark
SPARSE_BENCH_CLONE = $(SPARSE_BENCH_DIR)/.git
SPARSE_BENCH = deps/SparseRooflineBenchmark/build/hello

$(SPARSE_BENCH_CLONE): 
	git submodule update --init $(SPARSE_BENCH_DIR)

$(SPARSE_BENCH): $(SPARSE_BENCH_CLONE)
	mkdir -p $(SPARSE_BENCH) ;\
	touch $(SPARSE_BENCH)


TACO_DIR = deps/taco
TACO_CLONE = $(TACO_DIR)/.git
TACO = deps/taco/build/lib/libtaco.*
TACO_CXXFLAGS = -I$(TACO_DIR)/include -I$(TACO_DIR)/src
TACO_LDLIBS = -L$(TACO_DIR)/build/lib -ltaco -ldl

EIGEN_DIR = deps/eigen
EIGEN_CLONE = $(EIGEN_DIR)/.git
EIGEN_CXXFLAGS = -I$(EIGEN_DIR)

MKLROOT = deps/intel/mkl/2024.2
MKL_CXXFLAGS = -I$(MKLROOT)/include
MKL_LDLIBS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential

$(TACO_CLONE): 
	git submodule update --init $(TACO_DIR)

$(TACO): $(TACO_CLONE)
	cd $(TACO_DIR) ;\
	mkdir -p build ;\
	cd build ;\
	cmake -DPYTHON=false -DCMAKE_BUILD_TYPE=Release .. ;\
	make taco -j$(NPROC_VAL)

$(EIGEN_CLONE): 
	git submodule update --init $(EIGEN_DIR)

CORA_DIR = deps/cora
CORA_Z3 = $(CORA_DIR)/z3/hello
CORA_LLVM = $(CORA_DIR)/llvm/hello
CORA_CLONE = $(CORA_DIR)/.git
CORA = deps/cora/build/lib/libcora.*

$(CORA_CLONE):
	git submodule update --init $(CORA_DIR)

$(CORA_LLVM):
	cd $(CORA_DIR) ;\
	curl -L https://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-pc-linux-gnu.tar.xz -o llvm.tar.xz ;\
	tar -xf llvm.tar.xz ;\
	touch llvm/hello

$(CORA_Z3):
	cd $(CORA_DIR) ;\
	curl -L https://github.com/Z3Prover/z3/releases/download/z3-4.8.8/z3-4.8.8-x64-ubuntu-16.04.zip -o z3.zip :\
	unzip -q z3.zip ;\
	touch z3/hello

$(CORA): $(CORA_CLONE) $(CORA_LLVM) $(CORA_Z3)
	LLVM_PATH=$(CORA_DIR)/llvm ;\
	LD_LIBRARY_PATH=$(CORA_DIR)/z3/lib ;\
	CPATH=$(CORA_DIR)/z3/include ;\
	C_PLUS_INCLUDE_PATH=$(CORA_DIR)/z3/include ;\
	cd $(CORA_DIR) ;\
	mkdir build ;\
	cp config.cmake.arm build/config.cmake ;\
	cd build ;\
	cmake .. ;\
	make -j8 tvm

clean:
	rm -f $(ALL_TARGETS)
	rm -rf *.o *.dSYM *.trace

spgemm/spgemm_taco: $(SPARSE_BENCH) $(TACO) spgemm/spgemm_taco.cpp
	$(CXX) $(CXXFLAGS) $(TACO_CXXFLAGS) -o $@ spgemm/spgemm_taco.cpp $(LDLIBS) $(TACO_LDLIBS)

spgemm/spgemm_eigen: $(SPARSE_BENCH) $(EIGEN_CLONE) spgemm/spgemm_eigen.cpp
	$(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) -o $@ spgemm/spgemm_eigen.cpp

spmv/spmv_taco: $(SPARSE_BENCH) $(TACO) spmv/spmv_taco.cpp
	$(CXX) $(CXXFLAGS) $(TACO_CXXFLAGS) -o $@ spmv/spmv_taco.cpp $(LDLIBS) $(TACO_LDLIBS)

spmv/spmv_eigen: $(SPARSE_BENCH) $(EIGEN_CLONE) spmv/spmv_eigen.cpp
	$(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) -o $@ spmv/spmv_eigen.cpp

spmv/spmv_mkl: $(SPARSE_BENCH) spmv/spmv_mkl.cpp
	bash -c 'source deps/intel/setvars.sh; $(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) $(MKL_CXXFLAGS) -o $@ spmv/spmv_mkl.cpp $(LDLIBS) $(MKL_LDLIBS)'

spgemm/spgemm_mkl: $(SPARSE_BENCH) spgemm/spgemm_mkl.cpp
	bash -c 'source deps/intel/setvars.sh; $(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) $(MKL_CXXFLAGS) -o $@ spgemm/spgemm_mkl.cpp $(LDLIBS) $(MKL_LDLIBS)'
