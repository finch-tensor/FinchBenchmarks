#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

#matvec = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (j)>,    // b
    affine_map<(i,j) -> (i)>     // x (out)
  ],
  iterator_types = ["parallel", "reduction"],
  doc = "x(i) += A(i,j) * b(j)"
}

module {
  func.func @read_matrix(%fname: !llvm.ptr) -> tensor<?x?xf64, #CSR>
      attributes { llvm.emit_c_interface } {
    %t = sparse_tensor.new %fname : !llvm.ptr to tensor<?x?xf64, #CSR>
    return %t : tensor<?x?xf64, #CSR>
  }

  func.func @num_rows(%a: tensor<?x?xf64, #CSR>) -> i64
      attributes { llvm.emit_c_interface } {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %a, %c0 : tensor<?x?xf64, #CSR>
    %r = arith.index_cast %d : index to i64
    return %r : i64
  }

  func.func @spmv(%a: tensor<?x?xf64, #CSR>,
                  %x: tensor<?xf64>) -> tensor<?xf64>
      attributes { llvm.emit_c_interface } {
    %c1 = arith.constant 1 : index
    %one = arith.constant 1.0 : f64
    %n = tensor.dim %a, %c1 : tensor<?x?xf64, #CSR>
    %be = tensor.empty(%n) : tensor<?xf64>
    // b is all ones for testing case
    %b = linalg.fill ins(%one : f64) outs(%be : tensor<?xf64>) -> tensor<?xf64>
    %0 = linalg.generic #matvec
      ins(%a, %b : tensor<?x?xf64, #CSR>, tensor<?xf64>)
      outs(%x : tensor<?xf64>) {
        ^bb0(%av: f64, %bv: f64, %xv: f64):
          %m = arith.mulf %av, %bv : f64
          %s = arith.addf %xv, %m : f64
          linalg.yield %s : f64
      } -> tensor<?xf64>
    return %0 : tensor<?xf64>
  }
}

