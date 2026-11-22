module {
  // f(x) = c x^2, with c captured from outside the region: f'(x) = 2 c x.
  func.func @grad_scaled_square(%c: f64, %x: f64) -> f64 {
    %seed = arith.constant 1.0 : f64
    %grad = enzyme.autodiff_region(%x, %seed) {
    ^bb0(%a: f64):
      %sq = arith.mulf %a, %a : f64
      %r = arith.mulf %c, %sq : f64
      enzyme.yield %r : f64
    } <{activity = [#enzyme<activity enzyme_active>],
        ret_activity = [#enzyme<activity enzyme_activenoneed>]}> : (f64, f64) -> f64
    return %grad : f64
  }

  // f(x) = c sin(x), with c captured from outside the region: f'(x) dx = c cos(x) dx.
  func.func @jvp_scaled_sin(%c: f64, %x: f64, %dx: f64) -> f64 {
    %tangent = enzyme.fwddiff_region(%x, %dx) {
    ^bb0(%a: f64):
      %s = math.sin %a : f64
      %r = arith.mulf %c, %s : f64
      enzyme.yield %r : f64
    } <{activity = [#enzyme<activity enzyme_dup>],
        ret_activity = [#enzyme<activity enzyme_dupnoneed>]}> : (f64, f64) -> f64
    return %tangent : f64
  }
}
