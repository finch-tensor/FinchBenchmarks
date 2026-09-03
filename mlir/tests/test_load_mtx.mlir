#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
#DenseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : dense) }>

llvm.mlir.global internal constant @square_path("data/square.mtx\00")
llvm.mlir.global internal constant @wide_path("data/wide.mtx\00")
llvm.mlir.global internal constant @dense_path("data/dense.mtx\00")

func.func @load_csr_mtx(%path: !llvm.ptr) -> tensor<?x?xf64, #CSR> {
  %t = sparse_tensor.new %path : !llvm.ptr to tensor<?x?xf64, #CSR>
  return %t : tensor<?x?xf64, #CSR>
}

func.func @load_dense_vec_mtx(%path: !llvm.ptr) -> tensor<?xf64, #DenseVector> {
  %t = sparse_tensor.new %path : !llvm.ptr to tensor<?xf64, #DenseVector>
  return %t : tensor<?xf64, #DenseVector>
}

func.func @main() {
  // square 4x4 csr format with 5 nz
  %square_path = llvm.mlir.addressof @square_path : !llvm.ptr
  %square_csr = func.call @load_csr_mtx(%square_path) : (!llvm.ptr) -> tensor<?x?xf64, #CSR>
  sparse_tensor.print %square_csr : tensor<?x?xf64, #CSR>

  // wide 4x256 with 17 nz
  %wide_path = llvm.mlir.addressof @wide_path : !llvm.ptr
  %wide_csr = func.call @load_csr_mtx(%wide_path) : (!llvm.ptr) -> tensor<?x?xf64, #CSR>
  sparse_tensor.print %wide_csr : tensor<?x?xf64, #CSR>

  // 4x1 with 4 nz
  %dense_path = llvm.mlir.addressof @dense_path : !llvm.ptr
  %dense_vec = func.call @load_dense_vec_mtx(%dense_path) : (!llvm.ptr) -> tensor<?xf64, #DenseVector>
  sparse_tensor.print %dense_vec : tensor<?xf64, #DenseVector>
  return
}