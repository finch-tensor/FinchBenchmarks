CXX ?= g++

REPO_ROOT := $(abspath ../..)

HALIDE_DIR ?= $(shell cd $(REPO_ROOT) && poetry run python3 -c "import halide; print(halide.install_dir())" 2>/dev/null)
HALIDE_LIB := $(firstword $(wildcard $(HALIDE_DIR)/lib*/libHalide.so $(HALIDE_DIR)/lib*/libHalide.so.*))

CXXFLAGS += -std=c++17 -O3 -fPIC

NPROC_VAL ?= 4

all: halide_kernel halide_rfactor_kernel

OPENCV_DIR     = ../../deps/opencv
OPENCV_CLONE   = $(OPENCV_DIR)/.git
OPENCV_BUILD   = $(OPENCV_DIR)/build
OPENCV_INSTALL = $(abspath $(OPENCV_BUILD)/install)
OPENCV_STAMP   = $(OPENCV_INSTALL)/.built

$(OPENCV_CLONE):
	git submodule update --init $(OPENCV_DIR)

$(OPENCV_STAMP): $(OPENCV_CLONE)
	cd $(OPENCV_DIR) && \
	rm -rf build && \
	mkdir -p build && \
	cd build && \
	cmake \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_CXX_COMPILER=$(CXX) \
	    -DBUILD_LIST=core \
	    -DCMAKE_INSTALL_PREFIX=$(OPENCV_INSTALL) \
	    -DWITH_IPP=OFF \
	    -DWITH_OPENMP=OFF \
	    .. && \
	make -j$(NPROC_VAL) install
	touch $@

# halide-atomics: halide_impl.cpp, halide-rfactor: halide_rfactor_impl.cpp
halide_kernel: halide_impl.cpp $(OPENCV_STAMP)
halide_rfactor_kernel: halide_rfactor_impl.cpp $(OPENCV_STAMP)

halide_kernel halide_rfactor_kernel:
	@if [ -z "$(HALIDE_DIR)" ]; then echo "Error: Halide not installed (poetry install --no-root in $(REPO_ROOT))."; exit 1; fi
	@if [ -z "$(HALIDE_LIB)" ]; then echo "Error: libHalide not found under $(HALIDE_DIR)"; exit 1; fi
	$(CXX) \
	    $(CXXFLAGS) \
	    -I$(HALIDE_DIR)/include \
	    -I$(OPENCV_INSTALL)/include/opencv4 \
	    $< \
	    -o $@ \
	    $(LDFLAGS) \
	    $(HALIDE_LIB) \
	    -L$(OPENCV_INSTALL)/lib \
	    -L$(OPENCV_INSTALL)/lib64 \
	    -lopencv_core -ldl -lpthread \
	    -Wl,-rpath,$(dir $(HALIDE_LIB)) \
	    -Wl,-rpath,$(OPENCV_INSTALL)/lib \
	    -Wl,-rpath,$(OPENCV_INSTALL)/lib64

clean:
	rm -f halide_kernel halide_rfactor_kernel

clean-all: clean
	rm -rf $(OPENCV_BUILD)

.PHONY: all clean clean-all
