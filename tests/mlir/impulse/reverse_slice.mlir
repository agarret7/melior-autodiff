module {
  func.func @f(%x: tensor<4xf64>) -> tensor<2xf64> {
    %y = impulse.slice %x {start_indices = array<i64: 1>, limit_indices = array<i64: 3>,
                           strides = array<i64: 1>} : (tensor<4xf64>) -> tensor<2xf64>
    return %y : tensor<2xf64>
  }
  func.func @df(%x: tensor<4xf64>, %dr: tensor<2xf64>) -> tensor<4xf64> {
    %d = enzyme.autodiff @f(%x, %dr) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (tensor<4xf64>, tensor<2xf64>) -> tensor<4xf64>
    return %d : tensor<4xf64>
  }
}

// Impulse has no pad, so the adjoint writes dr into zeros at the static start index.
// CHECK-LABEL: func.func private @diffef(
// CHECK-SAME:    %{{[a-z0-9_]+}}: tensor<4xf64>, %[[DR:[a-z0-9_]+]]: tensor<2xf64>) -> tensor<4xf64>
// CHECK:         arith.addf %{{[0-9]+}}, %[[DR]]
// CHECK:         %[[DY:[0-9]+]] = "enzyme.get"
// CHECK:         %[[I:[a-z0-9_]+]] = arith.constant dense<1> : tensor<i64>
// CHECK:         %[[Z:[a-z0-9_]+]] = arith.constant dense<0.000000e+00> : tensor<4xf64>
// CHECK:         %[[DX:[0-9]+]] = impulse.dynamic_update_slice %[[Z]], %[[DY]], %[[I]] : (tensor<4xf64>, tensor<2xf64>, tensor<i64>) -> tensor<4xf64>
// CHECK:         arith.addf %{{[0-9]+}}, %[[DX]]
// CHECK:         return
