module {
  func.func @square(%x: f64) -> f64 {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func @jvp2(%x: f64, %d0: f64, %d1: f64, %out: memref<2xf64>)
      attributes { llvm.emit_c_interface } {
    %dx = tensor.from_elements %d0, %d1 : tensor<2xf64>
    %r = enzyme.fwddiff @square(%x, %dx) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dupnoneed>],
      width = 2
    } : (f64, tensor<2xf64>) -> tensor<2xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r0 = tensor.extract %r[%c0] : tensor<2xf64>
    %r1 = tensor.extract %r[%c1] : tensor<2xf64>
    memref.store %r0, %out[%c0] : memref<2xf64>
    memref.store %r1, %out[%c1] : memref<2xf64>
    return
  }
}
