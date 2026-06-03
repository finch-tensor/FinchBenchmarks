CC       = gcc
CXX      = g++
CXXFLAGS += -std=c++17 -O3 -fPIC -fopenmp

NPROC_VAL ?= 16

all: cv_kernel

OPENCV_DIR   = ../../deps/opencv
OPENCV_CLONE = $(OPENCV_DIR)/.git
OPENCV_BUILD = $(OPENCV_DIR)/build
OPENCV_LIB   = $(OPENCV_BUILD)/lib/libopencv_core.so


SPARSE_BENCH_DIR = ../../deps/SparseRooflineBenchmark
SPARSE_BENCH_CLONE = $(SPARSE_BENCH_DIR)/.git
SPARSE_BENCH = ../../deps/SparseRooflineBenchmark/build/hello

$(SPARSE_BENCH_CLONE):
	git submodule update --init $(SPARSE_BENCH_DIR)

$(SPARSE_BENCH): $(SPARSE_BENCH_CLONE)
	mkdir -p $(SPARSE_BENCH) ;\
	touch $(SPARSE_BENCH)


$(OPENCV_CLONE):
	git submodule update --init $(OPENCV_DIR)

$(OPENCV_LIB): $(OPENCV_CLONE)
	cd $(OPENCV_DIR) && \
	mkdir -p build && \
	cd build && \
	cmake \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DBUILD_LIST=core,imgproc,ximgproc \
	    -DCPU_BASELINE=SSE3 \
	    -DCPU_DISPATCH= \
	    -DCMAKE_INSTALL_PREFIX=$(PWD)/$(OPENCV_BUILD)/install \
	    .. && \
	make -j$(NPROC_VAL) && \
	make install

cv_kernel: cv_impl.cpp $(OPENCV_LIB)
	$(CXX) \
	    $(CXXFLAGS) \
	    -I$(OPENCV_DIR)/include \
	    -I$(OPENCV_DIR)/modules/core/include \
	    -I$(OPENCV_DIR)/modules/imgproc/include \
	    -I$(OPENCV_DIR)/modules/ximgproc/include \
	    -I$(OPENCV_BUILD) \
	    $< \
	    -o $@ \
	    -L$(OPENCV_BUILD)/lib \
		-Wl,-rpath,$(abspath $(OPENCV_BUILD)/lib) \
	    -lopencv_core \
	    -lopencv_imgproc


clean:
	rm -f cv_kernel

# clean-all: clean
# 	rm -rf $(TACO_DIR)/build