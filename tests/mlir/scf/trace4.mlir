module {
  func.func @trace4(%A: memref<4x4xf64>) -> f64
      attributes { llvm.emit_c_interface } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %z  = arith.constant 0.0 : f64
    %r  = scf.for %i = %c0 to %c4 step %c1 iter_args(%s = %z) -> f64 {
      %a  = memref.load %A[%i, %i] : memref<4x4xf64>
      %s2 = arith.addf %s, %a : f64
      scf.yield %s2 : f64
    }
    return %r : f64
  }
  func.func @jvp_trace4(%A: memref<4x4xf64>, %dA: memref<4x4xf64>) -> f64
      attributes { llvm.emit_c_interface } {
    %d = enzyme.fwddiff @trace4(%A, %dA) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dupnoneed>]
    } : (memref<4x4xf64>, memref<4x4xf64>) -> f64
    return %d : f64
  }
}
