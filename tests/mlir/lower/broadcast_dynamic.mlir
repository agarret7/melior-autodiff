module {
  func.func @broadcast(%x: memref<?xf64>, %out: memref<2x?xf64>) attributes { llvm.emit_c_interface } {
    %xt = bufferization.to_tensor %x restrict : memref<?xf64> to tensor<?xf64>
    %b = "enzyme.broadcast"(%xt) <{shape = array<i64: 2>}> : (tensor<?xf64>) -> tensor<2x?xf64>
    bufferization.materialize_in_destination %b in restrict writable %out
      : (tensor<2x?xf64>, memref<2x?xf64>) -> ()
    return
  }
}
