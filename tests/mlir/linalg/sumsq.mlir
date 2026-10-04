module {
  func.func @sumsq(%x: memref<4xf64>) -> f64 {
    %zero = arith.constant 0.0 : f64
    %acc = memref.alloca() : memref<f64>
    memref.store %zero, %acc[] : memref<f64>
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%x : memref<4xf64>) outs(%acc : memref<f64>) {
    ^bb0(%xi: f64, %a: f64):
      %sq = arith.mulf %xi, %xi : f64
      %s = arith.addf %a, %sq : f64
      linalg.yield %s : f64
    }
    %r = memref.load %acc[] : memref<f64>
    return %r : f64
  }
  func.func @vjp(%x: memref<4xf64>, %dx: memref<4xf64>, %dr: f64) {
    enzyme.autodiff @sumsq(%x, %dx, %dr) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (memref<4xf64>, memref<4xf64>, f64) -> ()
    return
  }
}

// Expected once Enzyme differentiates memref linalg.generic in reverse mode.
// CHECK-LABEL: func.func private @diffesumsq(
// CHECK:         enzyme.genericAdjoint
