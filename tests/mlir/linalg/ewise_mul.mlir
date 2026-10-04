module {
  func.func @ewise_mul(%x: tensor<4xf64>, %y: tensor<4xf64>) -> tensor<4xf64> {
    %empty = tensor.empty() : tensor<4xf64>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%x, %y : tensor<4xf64>, tensor<4xf64>) outs(%empty : tensor<4xf64>) {
    ^bb0(%xi: f64, %yi: f64, %out: f64):
      %prod = arith.mulf %xi, %yi : f64
      linalg.yield %prod : f64
    } -> tensor<4xf64>
    return %result : tensor<4xf64>
  }
  func.func @jvp_ewise_mul(%x: tensor<4xf64>, %dx: tensor<4xf64>,
                           %y: tensor<4xf64>, %dy: tensor<4xf64>) -> tensor<4xf64> {
    %d = enzyme.fwddiff @ewise_mul(%x, %dx, %y, %dy) {
      activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dupnoneed>]
    } : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %d : tensor<4xf64>
  }
}

// d(x·y) = dx·y + dy·x.
// CHECK-LABEL: func.func @jvp_ewise_mul(
// CHECK-NOT:     enzyme.fwddiff
// CHECK:         call @fwddiffeewise_mul(
// CHECK-LABEL: func.func private @fwddiffeewise_mul(
// CHECK:         linalg.generic
// CHECK-NEXT:    ^bb0(%[[X:[a-z0-9_]+]]: f64, %[[DX:[a-z0-9_]+]]: f64, %[[Y:[a-z0-9_]+]]: f64, %[[DY:[a-z0-9_]+]]: f64, %{{[a-z0-9_]+}}: f64, %{{[a-z0-9_]+}}: f64):
// CHECK-NEXT:      %[[A:[0-9]+]] = arith.mulf %[[DX]], %[[Y]]
// CHECK-NEXT:      %[[B:[0-9]+]] = arith.mulf %[[DY]], %[[X]]
// CHECK-NEXT:      %[[T:[0-9]+]] = arith.addf %[[A]], %[[B]]
// CHECK-NEXT:      %[[P:[0-9]+]] = arith.mulf %[[X]], %[[Y]]
// CHECK-NEXT:      linalg.yield %[[P]], %[[T]]
