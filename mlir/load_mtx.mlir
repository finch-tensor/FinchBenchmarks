#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
#DenseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : dense) }>

func.func @load_csr_mtx(%path: !llvm.ptr) -> tensor<?x?xf64, #CSR> {
  %t = sparse_tensor.new %path : !llvm.ptr to tensor<?x?xf64, #CSR>
  return %t : tensor<?x?xf64, #CSR>
}

func.func @load_dense_vec_mtx(%path: !llvm.ptr) -> tensor<?xf64, #DenseVector> {
  %t = sparse_tensor.new %path : !llvm.ptr to tensor<?xf64, #DenseVector>
  return %t : tensor<?xf64, #DenseVector>
}