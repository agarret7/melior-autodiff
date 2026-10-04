module {
  func.func @square(%x: f64) -> f64 {
    %y = math.sin %x : f64
    return %y : f64
  }
  func.func @dsquare(%x: tensor<4xf64>) -> tensor<4xf64> {
    %r = enzyme.batch @square(%x) { batch_shape = array<i64: 4> }
      : (tensor<4xf64>) -> tensor<4xf64>
    return %r : tensor<4xf64>
  }
}

// CHECK-LABEL: func.func @dsquare(
// CHECK-NOT:     enzyme.batch
// CHECK:         call @batched_square(%{{.*}}) : (tensor<4xf64>) -> tensor<4xf64>
// CHECK-LABEL: func.func private @batched_square(
// CHECK:         math.sin %{{.*}} : tensor<4xf64>
