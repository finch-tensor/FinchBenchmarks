CC       = gcc
CXX      = g++
CXXFLAGS += -std=c++17 -O3 -fPIC -fopenmp

# Auto-detect core count for parallel TACO build
ifeq ("$(shell uname)","Darwin")
    export NPROC_VAL := $(shell sysctl -n hw.logicalcpu_max)
else
    export NPROC_VAL := $(shell lscpu -p | egrep -v '^\#' | wc -l)
endif

TACO_DIR      = ../deps/taco
TACO_CLONE    = $(TACO_DIR)/.git
TACO_LIB      = $(TACO_DIR)/build/lib/libtaco.so
TACO_CXXFLAGS = $(CXXFLAGS) -I$(TACO_DIR)/include -I$(TACO_DIR)/src
TACO_LDLIBS   = -L$(TACO_DIR)/build/lib -ltaco -ldl \
                -Wl,-rpath,$(abspath $(TACO_DIR)/build/lib)

EIGEN_DIR      = ../deps/eigen
EIGEN_STAMP    = $(EIGEN_DIR)/.downloaded
EIGEN_CXXFLAGS = $(CXXFLAGS) -I$(EIGEN_DIR)


all: taco_kernel eigen_kernel mkl_kernel

SPARSE_BENCH_DIR = ../deps/SparseRooflineBenchmark
SPARSE_BENCH_CLONE = $(SPARSE_BENCH_DIR)/.git
SPARSE_BENCH = ../deps/SparseRooflineBenchmark/build/hello

$(SPARSE_BENCH_CLONE): 
	git submodule update --init $(SPARSE_BENCH_DIR)

$(SPARSE_BENCH): $(SPARSE_BENCH_CLONE)
	mkdir -p $(SPARSE_BENCH) ;\
	touch $(SPARSE_BENCH)


# Clone TACO submodule if missing
$(TACO_CLONE):
	git submodule update --init $(TACO_DIR)

$(TACO_LIB): $(TACO_CLONE)
	cd $(TACO_DIR) && \
	mkdir -p build && \
	cd build && \
	cmake -DPYTHON=false -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DOPENMP=ON .. && \
	make taco -j$(NPROC_VAL)

taco_kernel: taco_impl.cpp $(TACO_LIB) $(SPARSE_BENCH)
	$(CXX) $(TACO_CXXFLAGS) -o $@ $< $(TACO_LDLIBS)


# Clone Eigen submodule if missing
$(EIGEN_STAMP):
	mkdir -p $(EIGEN_DIR) && \
	curl -L https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.gz | \
	tar -xz --strip-components=1 -C $(EIGEN_DIR) && \
	touch $@

# Eigen is header-only
eigen_kernel: eigen_impl.cpp | $(EIGEN_STAMP) $(SPARSE_BENCH)
	$(CXX) $(TACO_CXXFLAGS) $(EIGEN_CXXFLAGS) -o $@ $< $(TACO_LDLIBS)


MKL_ROOT    ?= /usr
MKL_INCLUDE  = $(MKL_ROOT)/include/mkl
MKL_LIB      = $(MKL_ROOT)/lib/x86_64-linux-gnu

MKL_FLAGS   = -I$(MKL_INCLUDE) \
              -L$(MKL_LIB) \
              -lmkl_rt \
              -Wl,-rpath,$(MKL_LIB)

mkl_kernel: mkl_impl.cpp $(SPARSE_BENCH)
	$(CXX) $(TACO_CXXFLAGS) $(CXXFLAGS) $< -o $@ $(TACO_LDLIBS) $(MKL_FLAGS)


clean:
	rm -f taco_kernel eigen_kernel mkl_kernel

clean-all: clean
	rm -rf $(TACO_DIR)/build