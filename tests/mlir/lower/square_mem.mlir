module {
  func.func @square_mem(%x: f64) -> f64 {
    %buf = memref.alloca() : memref<f64>
    %sq = arith.mulf %x, %x : f64
    memref.store %sq, %buf[] : memref<f64>
    %r = memref.load %buf[] : memref<f64>
    return %r : f64
  }
  func.func @dsquare_mem(%x: f64, %dr: f64) -> f64 attributes { llvm.emit_c_interface } {
    %d = enzyme.autodiff @square_mem(%x, %dr) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64) -> f64
    return %d : f64
  }
}
