module {
  func.func @ewise_sq(%x: tensor<4xf64>) -> tensor<4xf64> {
    %empty = tensor.empty() : tensor<4xf64>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%x : tensor<4xf64>) outs(%empty : tensor<4xf64>) {
    ^bb0(%xi: f64, %out: f64):
      %sq = arith.mulf %xi, %xi : f64
      linalg.yield %sq : f64
    } -> tensor<4xf64>
    return %result : tensor<4xf64>
  }
  func.func @jvp_ewise_sq(%x: tensor<4xf64>, %dx: tensor<4xf64>) -> tensor<4xf64> {
    %d = enzyme.fwddiff @ewise_sq(%x, %dx) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dupnoneed>]
    } : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %d : tensor<4xf64>
  }
}

// d(x²) = dx·x + dx·x, computed alongside the primal in one doubled linalg.generic.
// CHECK-LABEL: func.func @jvp_ewise_sq(
// CHECK-NOT:     enzyme.fwddiff
// CHECK:         call @fwddiffeewise_sq(
// CHECK-LABEL: func.func private @fwddiffeewise_sq(
// CHECK:         linalg.generic
// CHECK-NEXT:    ^bb0(%[[X:[a-z0-9_]+]]: f64, %[[DX:[a-z0-9_]+]]: f64, %{{[a-z0-9_]+}}: f64, %{{[a-z0-9_]+}}: f64):
// CHECK-NEXT:      %[[A:[0-9]+]] = arith.mulf %[[DX]], %[[X]]
// CHECK-NEXT:      %[[B:[0-9]+]] = arith.mulf %[[DX]], %[[X]]
// CHECK-NEXT:      %[[T:[0-9]+]] = arith.addf %[[A]], %[[B]]
// CHECK-NEXT:      %[[P:[0-9]+]] = arith.mulf %[[X]], %[[X]]
// CHECK-NEXT:      linalg.yield %[[P]], %[[T]]
