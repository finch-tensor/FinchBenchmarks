CC  ?= clang
CXX ?= clang++

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    OMP_FLAGS = -Xpreprocessor -fopenmp
    OMP_LIBS  = -lomp
    OMP_INC   = $(shell brew --prefix libomp)/include
    OMP_LIB   = $(shell brew --prefix libomp)/lib
else
    OMP_FLAGS = -fopenmp
    OMP_LIBS  = -fopenmp
    OMP_INC   =
    OMP_LIB   =
endif

CXXFLAGS += \
    -std=c++17 -O3 -fPIC \
    $(OMP_FLAGS) \
    $(if $(OMP_INC),-I$(OMP_INC))

LDFLAGS += \
    $(OMP_LIBS) \
    $(if $(OMP_LIB),-L$(OMP_LIB))

NPROC_VAL ?= 4

all: cv_kernel

OPENCV_DIR   = ../../deps/opencv
OPENCV_CLONE = $(OPENCV_DIR)/.git
OPENCV_BUILD = $(OPENCV_DIR)/build
OPENCV_LIB   = $(OPENCV_BUILD)/lib/libopencv_core*
OPENCV_INSTALL = $(OPENCV_BUILD)/install


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
	rm -rf build && \
	mkdir -p build && \
	cd build && \
	cmake \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CXX_COMPILER=$(CXX) \
	    -DBUILD_LIST=core,imgproc,ximgproc \
	    -DCMAKE_INSTALL_PREFIX=$(PWD)/$(OPENCV_BUILD)/install \
	    .. && \
	make -j$(NPROC_VAL) && \
	make install

# cv_kernel: cv_impl.cpp $(OPENCV_LIB)
# 	$(CXX) \
# 	    $(CXXFLAGS) \
# 	    -I$(OPENCV_DIR)/include \
# 	    -I$(OPENCV_DIR)/modules/core/include \
# 	    -I$(OPENCV_DIR)/modules/imgproc/include \
# 	    -I$(OPENCV_DIR)/modules/ximgproc/include \
# 	    -I$(OPENCV_BUILD) \
# 	    $< \
# 	    -o $@ \
# 	    -L$(OPENCV_BUILD)/lib \
# 		-Wl,-rpath,$(abspath $(OPENCV_BUILD)/lib) \
# 	    -lopencv_core \
# 	    -lopencv_imgproc

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
	    -L$(OPENCV_INSTALL)/lib \
		-Wl,-rpath,$(OPENCV_INSTALL)/lib \
	    $(LDFLAGS) \
	    -lopencv_core \
	    -lopencv_imgproc

clean:
	rm -f cv_kernel

clean-all: clean
	rm -rf $(OPENCV_DIR)/build