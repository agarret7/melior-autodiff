module {
  func.func @f(%x: tensor<4xf64>) -> tensor<4xf64> {
    %i = arith.constant dense<1> : tensor<i64>
    %u = arith.constant dense<[7.0, 8.0]> : tensor<2xf64>
    %y = impulse.dynamic_update_slice %x, %u, %i
      : (tensor<4xf64>, tensor<2xf64>, tensor<i64>) -> tensor<4xf64>
    return %y : tensor<4xf64>
  }
  func.func @df(%x: tensor<4xf64>, %dr: tensor<4xf64>) -> tensor<4xf64> {
    %d = enzyme.autodiff @f(%x, %dr) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %d : tensor<4xf64>
  }
}

// The overwritten window of x gets no gradient: dx = dynamic_update_slice(dr, zeros, i).
// The update is a constant, so no gradient flows to it.
// CHECK-LABEL: func.func private @diffef(
// CHECK-SAME:    %{{[a-z0-9_]+}}: tensor<4xf64>, %[[DR:[a-z0-9_]+]]: tensor<4xf64>) -> tensor<4xf64>
// CHECK:         arith.addf %{{[0-9]+}}, %[[DR]]
// CHECK:         %[[DY:[0-9]+]] = "enzyme.get"
// CHECK:         %[[I:[0-9]+]] = "enzyme.pop"
// CHECK:         %[[Z:[a-z0-9_]+]] = arith.constant dense<0.000000e+00> : tensor<2xf64>
// CHECK:         %[[DX:[0-9]+]] = impulse.dynamic_update_slice %[[DY]], %[[Z]], %[[I]] : (tensor<4xf64>, tensor<2xf64>, tensor<i64>) -> tensor<4xf64>
// CHECK:         arith.addf %{{[0-9]+}}, %[[DX]]
// CHECK:         return
