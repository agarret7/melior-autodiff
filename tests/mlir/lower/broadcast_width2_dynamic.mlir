module {
  func.func private @square(%x: tensor<?xf64>) -> tensor<?xf64> {
    %y = arith.mulf %x, %x : tensor<?xf64>
    return %y : tensor<?xf64>
  }
  func.func @jvp2_dynamic(%x: memref<?xf64>, %dx: memref<2x?xf64>, %out: memref<2x?xf64>)
      attributes { llvm.emit_c_interface } {
    %xt = bufferization.to_tensor %x restrict : memref<?xf64> to tensor<?xf64>
    %dxt = bufferization.to_tensor %dx restrict : memref<2x?xf64> to tensor<2x?xf64>
    %r = enzyme.fwddiff @square(%xt, %dxt) <{
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dupnoneed>],
      width = 2
    }> : (tensor<?xf64>, tensor<2x?xf64>) -> tensor<2x?xf64>
    bufferization.materialize_in_destination %r in restrict writable %out
      : (tensor<2x?xf64>, memref<2x?xf64>) -> ()
    return
  }
}
