#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
#DenseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : dense) }>

module {
  func.func @spmv(%A: tensor<?x?xf64, #CSR>, %b: tensor<?xf64, #DenseVector>, %x: memref<?xf64>)
    attributes {llvm.emit_c_interface}
  {
    // TODO: does this actually get allocated or optimized out?
    %x_t = bufferization.to_tensor %x restrict writable : memref<?xf64> to tensor<?xf64>
    %out = linalg.matvec
      ins(%A, %b : tensor<?x?xf64, #CSR>, tensor<?xf64, #DenseVector>)
      outs(%x_t : tensor<?xf64>) -> tensor<?xf64>
    bufferization.materialize_in_destination %out in restrict writable %x : (tensor<?xf64>, memref<?xf64>) -> ()
    return
  }
}