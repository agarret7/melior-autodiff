module {
  func.func private @square(%x: f64) -> f64 {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func @square4(%x: memref<4xf64>, %out: memref<4xf64>) attributes { llvm.emit_c_interface } {
    %xt = bufferization.to_tensor %x restrict : memref<4xf64> to tensor<4xf64>
    %r = enzyme.batch @square(%xt) <{ batch_shape = array<i64: 4> }>
      : (tensor<4xf64>) -> tensor<4xf64>
    bufferization.materialize_in_destination %r in restrict writable %out
      : (tensor<4xf64>, memref<4xf64>) -> ()
    return
  }
}
