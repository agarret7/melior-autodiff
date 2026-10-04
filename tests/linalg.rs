use melior::ir::Module;
use melior::pass::PassManager;
use melior_autodiff::{create_context, enzymeCreateDifferentiatePass};

mod filecheck;
use filecheck::filecheck;

// Differentiates tests/<file> and FileChecks the result against the CHECK lines in it.
fn differentiate(file: &str) {
    let src = std::fs::read_to_string(format!("{}/tests/{file}", env!("CARGO_MANIFEST_DIR")))
        .expect("missing test file");
    let ctx = create_context();
    let mut module = Module::parse(&ctx, &src).expect("parse failed");
    let pm = PassManager::new(&ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });
    pm.run(&mut module).expect("differentiate pass failed");
    filecheck(&module.as_operation().to_string(), file, None);
}

// f(x)_i = x_i^2.  JVP tangent: (df)_i = 2 * x_i * (dx)_i.
#[test]
fn fwddiff_linalg_generic_ewise_sq() {
    differentiate("mlir/linalg/ewise_sq.mlir");
}

// trace(A) = sum_i A[i,i].  JVP: fwddiff(@trace4, A, dA) = trace(dA).
// Uses the diagonal indexing map (i) -> (i, i) — a non-trivial linalg.generic access pattern.
#[test]
fn fwddiff_linalg_generic_trace() {
    differentiate("mlir/linalg/trace4_tensor.mlir");
}

// f(x, y)_i = x_i * y_i.  JVP: (df)_i = x_i*(dy)_i + y_i*(dx)_i.
#[test]
fn fwddiff_linalg_generic_ewise_mul() {
    differentiate("mlir/linalg/ewise_mul.mlir");
}

// f(x) = Σ xᵢ² over a memref.  Reverse mode on tensors is unsupported in Enzyme, and on memrefs
// the linalg.generic is silently skipped: no enzyme.genericAdjoint is emitted, so ∇f comes back 0.
#[test]
#[ignore = "Enzyme skips reverse-mode linalg.generic on memrefs; gradient is silently zero"]
fn autodiff_linalg_generic_reduction_memref() {
    let ir = differentiate(
        r#"
module {
  func.func @sumsq(%x: memref<4xf64>) -> f64 {
    %zero = arith.constant 0.0 : f64
    %acc = memref.alloca() : memref<f64>
    memref.store %zero, %acc[] : memref<f64>
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%x : memref<4xf64>) outs(%acc : memref<f64>) {
    ^bb0(%xi: f64, %a: f64):
      %sq = arith.mulf %xi, %xi : f64
      %s = arith.addf %a, %sq : f64
      linalg.yield %s : f64
    }
    %r = memref.load %acc[] : memref<f64>
    return %r : f64
  }
  func.func @vjp(%x: memref<4xf64>, %dx: memref<4xf64>, %dr: f64) {
    enzyme.autodiff @sumsq(%x, %dx, %dr) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (memref<4xf64>, memref<4xf64>, f64) -> ()
    return
  }
}
"#,
    );
    assert!(ir.contains("diffesumsq"), "no reverse function generated:\n{ir}");
    assert!(ir.contains("enzyme.genericAdjoint"), "linalg.generic has no adjoint:\n{ir}");
}
