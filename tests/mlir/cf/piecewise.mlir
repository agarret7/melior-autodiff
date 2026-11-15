module {
  // f(x) = x² if x > 0, else -x: a diamond of blocks joined by a block argument.
  func.func private @piecewise(%x: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %pos = arith.cmpf ogt, %x, %zero : f64
    cf.cond_br %pos, ^square, ^negate
  ^square:
    %sq = arith.mulf %x, %x : f64
    cf.br ^join(%sq : f64)
  ^negate:
    %neg = arith.negf %x : f64
    cf.br ^join(%neg : f64)
  ^join(%r: f64):
    return %r : f64
  }
  func.func @fwd_piecewise(%x: f64, %dx: f64) -> f64 {
    %d = enzyme.fwddiff @piecewise(%x, %dx) <{activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>]}> : (f64, f64) -> f64
    return %d : f64
  }
  func.func @rev_piecewise(%x: f64, %dr: f64) -> f64 {
    %d = enzyme.autodiff @piecewise(%x, %dr) <{activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]}> : (f64, f64) -> f64
    return %d : f64
  }
}
