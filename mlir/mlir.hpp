#include <cstdint>

// https://discourse.llvm.org/t/how-to-call-mlir-function-by-c/73126
template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

extern "C" void *load_csr_mtx(const char *path);
extern "C" void *load_dense_vec_mtx(const char *path);
extern "C" void print_csr(void *A);
extern "C" void output_csr(void *A, const char *path);