module {
  // 9 steps of s <- cos(s^2) from s = x, with binomial checkpointing: Enzyme keeps 3 snapshots of
  // s and recomputes the steps between them in the reverse pass, instead of caching every step.
  func.func @iterate_checkpointed(%x: f64) -> f64 {
    %lb = arith.constant 0 : index
    %ub = arith.constant 9 : index
    %step = arith.constant 1 : index
    %s = scf.for %i = %lb to %ub step %step iter_args(%acc = %x) -> (f64) {
      %sq = arith.mulf %acc, %acc : f64
      %next = math.cos %sq : f64
      scf.yield %next : f64
    } {enzyme.enable_checkpointing = true, enzyme.binomial_checkpointing,
       enzyme.checkpoint_period = 3 : i64}
    return %s : f64
  }

  // The same loop without checkpointing.
  func.func @iterate(%x: f64) -> f64 {
    %lb = arith.constant 0 : index
    %ub = arith.constant 9 : index
    %step = arith.constant 1 : index
    %s = scf.for %i = %lb to %ub step %step iter_args(%acc = %x) -> (f64) {
      %sq = arith.mulf %acc, %acc : f64
      %next = math.cos %sq : f64
      scf.yield %next : f64
    }
    return %s : f64
  }

  func.func @grad_iterate_checkpointed(%x: f64, %dr: f64) -> f64 {
    %r = enzyme.autodiff @iterate_checkpointed(%x, %dr) <{
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    }> : (f64, f64) -> f64
    return %r : f64
  }

  func.func @grad_iterate(%x: f64, %dr: f64) -> f64 {
    %r = enzyme.autodiff @iterate(%x, %dr) <{
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    }> : (f64, f64) -> f64
    return %r : f64
  }
}
