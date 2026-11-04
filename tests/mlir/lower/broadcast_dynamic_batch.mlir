module {
  func.func @broadcast(%x: tensor<3xf64>) -> tensor<?x3xf64> {
    %b = "enzyme.broadcast"(%x) <{shape = array<i64: 2>}> : (tensor<3xf64>) -> tensor<?x3xf64>
    return %b : tensor<?x3xf64>
  }
}
