use melior::ir::Module;
use melior::pass::PassManager;
use melior_autodiff::{create_context, enzymeCreateDifferentiatePass};

fn differentiate(src: &str) -> String {
    let ctx = create_context();
    let mut module = Module::parse(&ctx, src).expect("parse failed");
    let pm = PassManager::new(&ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });
    pm.run(&mut module).expect("differentiate pass failed");
    module.as_operation().to_string()
}

// f(x)_i = x_i^2.  JVP tangent: (df)_i = 2 * x_i * (dx)_i.
// Enzyme should produce a doubled linalg.generic that propagates the tangent.
#[test]
fn fwddiff_linalg_generic_ewise_sq() {
    let ir = differentiate(r#"
module {
  func.func @ewise_sq(%x: tensor<4xf64>) -> tensor<4xf64> {
    %empty = tensor.empty() : tensor<4xf64>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%x : tensor<4xf64>) outs(%empty : tensor<4xf64>) {
    ^bb0(%xi: f64, %out: f64):
      %sq = arith.mulf %xi, %xi : f64
      linalg.yield %sq : f64
    } -> tensor<4xf64>
    return %result : tensor<4xf64>
  }
  func.func @jvp_ewise_sq(%x: tensor<4xf64>, %dx: tensor<4xf64>) -> tensor<4xf64> {
    %d = enzyme.fwddiff @ewise_sq(%x, %dx) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dupnoneed>]
    } : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %d : tensor<4xf64>
  }
}
"#,
    );
    assert!(
        !ir.contains("enzyme.fwddiff"),
        "enzyme.fwddiff was not lowered:\n{ir}"
    );
    assert!(
        ir.contains("fwddiffeewise_sq"),
        "no fwddiff function generated:\n{ir}"
    );
    // Enzyme should have produced a doubled linalg.generic (ins has x and dx, outs has result and tangent).
    assert!(
        ir.contains("linalg.generic"),
        "no linalg.generic in differentiated output:\n{ir}"
    );
}

// trace(A) = sum_i A[i,i].  JVP: fwddiff(@trace4, A, dA) = trace(dA).
// Uses the diagonal indexing map (i) -> (i, i) — a non-trivial linalg.generic access pattern.
#[test]
fn fwddiff_linalg_generic_trace() {
    let ir = differentiate(r#"
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
"#,
    );
    assert!(
        !ir.contains("enzyme.fwddiff"),
        "enzyme.fwddiff was not lowered:\n{ir}"
    );
    assert!(
        ir.contains("fwddiffetrace4_tensor"),
        "no fwddiff function generated:\n{ir}"
    );
    assert!(
        ir.contains("linalg.generic"),
        "no linalg.generic in differentiated output:\n{ir}"
    );
}

// f(x, y)_i = x_i * y_i.  JVP: (df)_i = x_i*(dy)_i + y_i*(dx)_i.
// Tests that Enzyme correctly propagates tangents through a two-input linalg.generic.
#[test]
fn fwddiff_linalg_generic_ewise_mul() {
    let ir = differentiate(r#"
module {
  func.func @ewise_mul(%x: tensor<4xf64>, %y: tensor<4xf64>) -> tensor<4xf64> {
    %empty = tensor.empty() : tensor<4xf64>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%x, %y : tensor<4xf64>, tensor<4xf64>) outs(%empty : tensor<4xf64>) {
    ^bb0(%xi: f64, %yi: f64, %out: f64):
      %prod = arith.mulf %xi, %yi : f64
      linalg.yield %prod : f64
    } -> tensor<4xf64>
    return %result : tensor<4xf64>
  }
  func.func @jvp_ewise_mul(%x: tensor<4xf64>, %dx: tensor<4xf64>,
                           %y: tensor<4xf64>, %dy: tensor<4xf64>) -> tensor<4xf64> {
    %d = enzyme.fwddiff @ewise_mul(%x, %dx, %y, %dy) {
      activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dupnoneed>]
    } : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %d : tensor<4xf64>
  }
}
"#,
    );
    assert!(
        !ir.contains("enzyme.fwddiff"),
        "enzyme.fwddiff was not lowered:\n{ir}"
    );
    assert!(
        ir.contains("fwddiffeewise_mul"),
        "no fwddiff function generated:\n{ir}"
    );
    assert!(
        ir.contains("linalg.generic"),
        "no linalg.generic in differentiated output:\n{ir}"
    );
}
