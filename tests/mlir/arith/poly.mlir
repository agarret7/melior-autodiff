module {
  func.func @poly(%x: f64) -> f64 {
    %c3 = arith.constant 3.0 : f64
    %c1 = arith.constant 1.0 : f64
    %x2 = arith.mulf %x, %x : f64
    %t  = arith.mulf %c3, %x : f64
    %y  = arith.subf %x2, %t : f64
    %y1 = arith.addf %y, %c1 : f64
    return %y1 : f64
  }
  func.func @dpoly(%x: f64, %dr: f64) -> f64 attributes { llvm.emit_c_interface } {
    %d = enzyme.autodiff @poly(%x, %dr) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64) -> f64
    return %d : f64
  }
}
