#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

llvm.mlir.global internal constant @square_path("data/square.mtx\00")
llvm.mlir.global internal constant @wide_path("data/wide.mtx\00")

func.func @load_mtx(%path: !llvm.ptr) -> tensor<?x?xf64, #CSR> {
  %t = sparse_tensor.new %path : !llvm.ptr to tensor<?x?xf64, #CSR>
  return %t : tensor<?x?xf64, #CSR>
}

func.func @main() {
  %square_path = llvm.mlir.addressof @square_path : !llvm.ptr
  %wide_path = llvm.mlir.addressof @wide_path : !llvm.ptr
  %square_csr = func.call @load_mtx(%square_path) : (!llvm.ptr) -> tensor<?x?xf64, #CSR>
  sparse_tensor.print %square_csr : tensor<?x?xf64, #CSR>
  %wide_csr = func.call @load_mtx(%wide_path) : (!llvm.ptr) -> tensor<?x?xf64, #CSR>
  sparse_tensor.print %wide_csr : tensor<?x?xf64, #CSR>
  bufferization.dealloc_tensor %square_csr : tensor<?x?xf64, #CSR>
  return
}