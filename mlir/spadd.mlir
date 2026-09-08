#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

func.func @spdelete(%C: tensor<?x?xf64, #CSR>) attributes {llvm.emit_c_interface}
{
  bufferization.dealloc_tensor %C : tensor<?x?xf64, #CSR>
  return
}

func.func @spadd(%A: tensor<?x?xf64, #CSR>, %B: tensor<?x?xf64, #CSR>) -> tensor<?x?xf64, #CSR> attributes {llvm.emit_c_interface}
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %A, %c0 : tensor<?x?xf64, #CSR>
  %d1 = tensor.dim %A, %c1 : tensor<?x?xf64, #CSR>
  %C = tensor.empty(%d0, %d1) : tensor<?x?xf64, #CSR>
  %R = linalg.add
         ins(%A, %B : tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>)
         outs(%C : tensor<?x?xf64, #CSR>) -> tensor<?x?xf64, #CSR>
  return %R : tensor<?x?xf64, #CSR>
}
