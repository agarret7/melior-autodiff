module {
  func.func @broadcast(%x: tensor<?xf64>) -> tensor<2x?xf64> {
    %b = "enzyme.broadcast"(%x) <{shape = array<i64: 2>}> : (tensor<?xf64>) -> tensor<2x?xf64>
    return %b : tensor<2x?xf64>
  }
}
