module {
  func.func @cube(%x: f64) -> f64 {
    %x2 = arith.mulf %x, %x : f64
    %x3 = arith.mulf %x2, %x : f64
    return %x3 : f64
  }
  func.func @dcube(%x: f64, %dr: f64) -> f64 attributes { llvm.emit_c_interface } {
    %d = enzyme.autodiff @cube(%x, %dr) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64) -> f64
    return %d : f64
  }
}
