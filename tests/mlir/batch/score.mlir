// The score ∇log p of a two-component Gaussian mixture, batched over 8 points.
// enzyme-batch crashes on it today: it asserts on scalar constants like `arith.constant 0.5 : f64`.
module {
  func.func private @logp(%x: f64) -> f64 {
    %half = arith.constant 0.5 : f64
    %mu = arith.constant 2.0 : f64
    %a = arith.subf %x, %mu : f64
    %b = arith.addf %x, %mu : f64
    %a2 = arith.mulf %a, %a : f64
    %b2 = arith.mulf %b, %b : f64
    %ea = arith.mulf %a2, %half : f64
    %eb = arith.mulf %b2, %half : f64
    %na = arith.negf %ea : f64
    %nb = arith.negf %eb : f64
    %pa = math.exp %na : f64
    %pb = math.exp %nb : f64
    %s = arith.addf %pa, %pb : f64
    %l = math.log %s : f64
    return %l : f64
  }
  func.func private @score(%x: f64) -> f64 {
    %one = arith.constant 1.0 : f64
    %g = enzyme.autodiff @logp(%x, %one) <{
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    }> : (f64, f64) -> f64
    return %g : f64
  }
  func.func @scores(%xs: memref<8xf64>, %out: memref<8xf64>) {
    %xt = bufferization.to_tensor %xs restrict : memref<8xf64> to tensor<8xf64>
    %g = enzyme.batch @score(%xt) <{ batch_shape = array<i64: 8> }> : (tensor<8xf64>) -> tensor<8xf64>
    bufferization.materialize_in_destination %g in restrict writable %out
      : (tensor<8xf64>, memref<8xf64>) -> ()
    return
  }
}
