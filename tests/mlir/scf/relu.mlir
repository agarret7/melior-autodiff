module {
  func.func @relu(%x: f64) -> f64 attributes { llvm.emit_c_interface } {
    %zero = arith.constant 0.0 : f64
    %cond = arith.cmpf ogt, %x, %zero : f64
    %result = scf.if %cond -> f64 {
      scf.yield %x : f64
    } else {
      scf.yield %zero : f64
    }
    return %result : f64
  }
  func.func @jvp_relu(%x: f64, %dx: f64) -> f64 attributes { llvm.emit_c_interface } {
    %d = enzyme.fwddiff @relu(%x, %dx) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dupnoneed>]
    } : (f64, f64) -> f64
    return %d : f64
  }
}
