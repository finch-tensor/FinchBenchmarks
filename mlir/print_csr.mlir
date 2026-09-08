#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1: dense, d0: compressed)
}>

func.func @print_csr(%A: tensor<?x?xf64, #CSR>)
{
    sparse_tensor.print %A: tensor<?x?xf64, #CSR>
    return
}

func.func @output_csr(%A: tensor<?x?xf64, #CSR>, %path: !llvm.ptr) {
  sparse_tensor.out %A, %path : tensor<?x?xf64, #CSR>, !llvm.ptr
  return
}
