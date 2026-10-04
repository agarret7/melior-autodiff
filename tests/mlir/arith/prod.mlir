module {
  func.func @prod(%x: f64, %y: f64) -> f64 {
    %r = arith.mulf %x, %y : f64
    return %r : f64
  }
  func.func @dprod_dx(%x: f64, %y: f64, %dr: f64) -> f64 attributes { llvm.emit_c_interface } {
    %d = enzyme.autodiff @prod(%x, %y, %dr) {
      activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64, f64) -> f64
    return %d : f64
  }
}
