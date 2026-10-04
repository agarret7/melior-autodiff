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
