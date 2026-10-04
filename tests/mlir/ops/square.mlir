// Input for the op-builder tests in tests/ops.rs, which append a @caller built through the C API.
// Each test checks its own prefix.
module {
  func.func @square(%x: f64) -> f64 {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
}

// AUTODIFF-LABEL: func.func @caller(
// AUTODIFF-SAME:          %[[X:[a-z0-9_]+]]: f64, %[[DR:[a-z0-9_]+]]: f64) -> f64
// AUTODIFF-NEXT:    enzyme.autodiff @square(%[[X]], %[[DR]]) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (f64, f64) -> f64

// FWDDIFF-LABEL: func.func @caller(
// FWDDIFF-SAME:         %[[X:[a-z0-9_]+]]: f64, %[[DX:[a-z0-9_]+]]: f64) -> f64
// FWDDIFF-NEXT:    enzyme.fwddiff @square(%[[X]], %[[DX]]) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>]} : (f64, f64) -> f64

// JACOBIAN-LABEL: func.func @caller(
// JACOBIAN-SAME:          %[[X:[a-z0-9_]+]]: f64, %[[DX:[a-z0-9_]+]]: f64) -> f64
// JACOBIAN-NEXT:    enzyme.jacobian @square(%[[X]], %[[DX]]) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>], strong_zero = true, width = 2 : i64} : (f64, f64) -> f64

// BATCH-LABEL: func.func @caller(
// BATCH-SAME:       %[[X:[a-z0-9_]+]]: tensor<4xf64>) -> tensor<4xf64>
// BATCH-NEXT:    enzyme.batch @square(%[[X]]) {batch_shape = array<i64: 4>} : (tensor<4xf64>) -> tensor<4xf64>

// Reverse mode of x²: x is cached twice and each copy is scaled by the same incoming gradient.
// LOWERED-LABEL: func.func @caller(
// LOWERED-NOT:     enzyme.autodiff
// LOWERED:         call @diffesquare(%arg0, %arg1) : (f64, f64) -> f64
// LOWERED-LABEL: func.func private @diffesquare(
// LOWERED-SAME:         %[[X:[a-z0-9_]+]]: f64, %[[DR:[a-z0-9_]+]]: f64) -> f64
// LOWERED:         "enzyme.push"(%{{[0-9]+}}, %[[X]])
// LOWERED:         "enzyme.push"(%{{[0-9]+}}, %[[X]])
// LOWERED:         arith.addf %{{[0-9]+}}, %[[DR]]
// LOWERED:         %[[DY:[0-9]+]] = "enzyme.get"
// LOWERED:         %[[X1:[0-9]+]] = "enzyme.pop"
// LOWERED:         %[[X2:[0-9]+]] = "enzyme.pop"
// LOWERED:         arith.mulf %[[DY]], %[[X2]]
// LOWERED:         arith.mulf %[[DY]], %[[X1]]

// No pass lowers enzyme.jacobian yet.
// JACOBIAN_LOWERED-LABEL: func.func @caller(
// JACOBIAN_LOWERED-NOT:     enzyme.jacobian
