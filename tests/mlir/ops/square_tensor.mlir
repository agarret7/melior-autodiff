// Input for differentiate_pass_lowers_tensor_autodiff in tests/ops.rs, which appends @caller.
module {
  func.func @square_tensor(%x: tensor<2xf64>) -> tensor<2xf64> {
    %y = arith.mulf %x, %x : tensor<2xf64>
    return %y : tensor<2xf64>
  }
}

// Same adjoint as the scalar case, with a tensor-valued gradient accumulator.
// CHECK-LABEL: func.func @caller(
// CHECK-NOT:     enzyme.autodiff
// CHECK:         call @diffesquare_tensor(
// CHECK-LABEL: func.func private @diffesquare_tensor(
// CHECK-SAME:    %{{[a-z0-9_]+}}: tensor<2xf64>, %[[DR:[a-z0-9_]+]]: tensor<2xf64>) -> tensor<2xf64>
// CHECK:         "enzyme.init"() : () -> !enzyme.Gradient<tensor<2xf64>>
// CHECK:         arith.addf %{{[0-9]+}}, %[[DR]]
// CHECK:         %[[DY:[0-9]+]] = "enzyme.get"
// CHECK:         %[[X1:[0-9]+]] = "enzyme.pop"
// CHECK:         %[[X2:[0-9]+]] = "enzyme.pop"
// CHECK:         arith.mulf %[[DY]], %[[X2]] fastmath<fast> : tensor<2xf64>
// CHECK:         arith.mulf %[[DY]], %[[X1]] fastmath<fast> : tensor<2xf64>
