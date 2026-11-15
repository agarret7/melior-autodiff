module {
  // x^n by repeated multiplication, as a loop of blocks with a back edge.
  func.func private @power(%x: f64, %n: i64) -> f64 {
    %one = arith.constant 1.0 : f64
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    cf.br ^loop(%c0, %one : i64, f64)
  ^loop(%i: i64, %acc: f64):
    %more = arith.cmpi slt, %i, %n : i64
    cf.cond_br %more, ^body, ^exit(%acc : f64)
  ^body:
    %next = arith.mulf %acc, %x : f64
    %i1 = arith.addi %i, %c1 : i64
    cf.br ^loop(%i1, %next : i64, f64)
  ^exit(%r: f64):
    return %r : f64
  }
  func.func @fwd_power(%x: f64, %dx: f64, %n: i64) -> f64 {
    %d = enzyme.fwddiff @power(%x, %dx, %n) <{activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>], ret_activity = [#enzyme<activity enzyme_dupnoneed>]}> : (f64, f64, i64) -> f64
    return %d : f64
  }
  func.func @rev_power(%x: f64, %n: i64, %dr: f64) -> f64 {
    %d = enzyme.autodiff @power(%x, %n, %dr) <{activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>], ret_activity = [#enzyme<activity enzyme_activenoneed>]}> : (f64, i64, f64) -> f64
    return %d : f64
  }
}
