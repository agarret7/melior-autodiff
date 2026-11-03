// Input for tests/wrap.rs, which runs enzyme-wrap on @square. Each test checks its own prefix.
module {
  func.func @square(%x: f64) -> f64 {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
}

// Reverse mode: d(x * x) = dr * x + dr * x.
// REVERSE-LABEL: func.func private @dsquare(
// REVERSE-SAME:         %[[X:[a-z0-9_]+]]: f64, %[[DR:[a-z0-9_]+]]: f64) -> f64
// REVERSE:         %[[A:[0-9]+]] = arith.mulf %[[DR]], %[[X]]
// REVERSE:         %[[B:[0-9]+]] = arith.mulf %[[DR]], %[[X]]
// REVERSE:         %[[G:[0-9]+]] = arith.addf %[[A]], %[[B]]
// REVERSE:         return %[[G]] : f64

// Forward mode, with the result Dup: returns x * x and its tangent, dx * x + dx * x.
// FORWARD-LABEL: func.func private @fwd_square(
// FORWARD-SAME:         %[[X:[a-z0-9_]+]]: f64, %[[DX:[a-z0-9_]+]]: f64) -> (f64, f64)
// FORWARD:         %[[A:[0-9]+]] = arith.mulf %[[DX]], %[[X]]
// FORWARD:         %[[B:[0-9]+]] = arith.mulf %[[DX]], %[[X]]
// FORWARD:         %[[T:[0-9]+]] = arith.addf %[[A]], %[[B]]
// FORWARD:         %[[Y:[0-9]+]] = arith.mulf %[[X]], %[[X]]
// FORWARD:         return %[[Y]], %[[T]] : f64, f64
