module {
  func.func @f(%x: tensor<4xf64>) -> tensor<2x2xf64> {
    %y = impulse.reshape %x : (tensor<4xf64>) -> tensor<2x2xf64>
    return %y : tensor<2x2xf64>
  }
  func.func @df(%x: tensor<4xf64>, %dr: tensor<2x2xf64>) -> tensor<4xf64> {
    %d = enzyme.autodiff @f(%x, %dr) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (tensor<4xf64>, tensor<2x2xf64>) -> tensor<4xf64>
    return %d : tensor<4xf64>
  }
}

// dx = reshape(dr) back to x's shape.
// CHECK-LABEL: func.func private @diffef(
// CHECK-SAME:    %{{[a-z0-9_]+}}: tensor<4xf64>, %[[DR:[a-z0-9_]+]]: tensor<2x2xf64>) -> tensor<4xf64>
// CHECK:         arith.addf %{{[0-9]+}}, %[[DR]]
// CHECK:         %[[DY:[0-9]+]] = "enzyme.get"
// CHECK:         %[[DX:[0-9]+]] = impulse.reshape %[[DY]] : (tensor<2x2xf64>) -> tensor<4xf64>
// CHECK:         arith.addf %{{[0-9]+}}, %[[DX]]
// CHECK:         return
