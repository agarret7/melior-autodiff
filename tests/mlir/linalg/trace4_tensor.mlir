module {
  func.func @trace4_tensor(%A: tensor<4x4xf64>) -> tensor<f64> {
    %zero = arith.constant 0.0 : f64
    %init = tensor.from_elements %zero : tensor<f64>
    %result = linalg.generic {
      indexing_maps = [affine_map<(i) -> (i, i)>,
                       affine_map<(i) -> ()>],
      iterator_types = ["reduction"]
    } ins(%A : tensor<4x4xf64>) outs(%init : tensor<f64>) {
    ^bb0(%a: f64, %acc: f64):
      %sum = arith.addf %acc, %a : f64
      linalg.yield %sum : f64
    } -> tensor<f64>
    return %result : tensor<f64>
  }
  func.func @jvp_trace4(%A: tensor<4x4xf64>, %dA: tensor<4x4xf64>) -> tensor<f64> {
    %d = enzyme.fwddiff @trace4_tensor(%A, %dA) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dupnoneed>]
    } : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<f64>
    return %d : tensor<f64>
  }
}

// d trace(A) = trace(dA): the tangent accumulates dA along the same diagonal map as A.
// CHECK:       #[[DIAG:map[0-9]*]] = affine_map<(d0) -> (d0, d0)>
// CHECK-LABEL: func.func @jvp_trace4(
// CHECK-NOT:     enzyme.fwddiff
// CHECK:         call @fwddiffetrace4_tensor(
// CHECK-LABEL: func.func private @fwddiffetrace4_tensor(
// CHECK:         linalg.generic {indexing_maps = [#[[DIAG]], #[[DIAG]],
// CHECK-NEXT:    ^bb0(%[[A:[a-z0-9_]+]]: f64, %[[DA:[a-z0-9_]+]]: f64, %[[ACC:[a-z0-9_]+]]: f64, %[[DACC:[a-z0-9_]+]]: f64):
// CHECK-NEXT:      %[[T:[0-9]+]] = arith.addf %[[DACC]], %[[DA]]
// CHECK-NEXT:      %[[P:[0-9]+]] = arith.addf %[[ACC]], %[[A]]
// CHECK-NEXT:      linalg.yield %[[P]], %[[T]]
