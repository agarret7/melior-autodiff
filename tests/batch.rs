use melior::ir::{Location, Module};
use melior::pass::PassManager;
use melior_autodiff::{pass, utility::create_context};

mod common;
use common::{compile, lookup};
mod filecheck;
use filecheck::filecheck;

// Runs enzyme-batch on a parsed module and checks that the batch op is lowered
// to a call to a generated @batched_* function.
#[test]
fn batch_pass_lowers_batch_op() {
    let ctx = create_context();
    let mut module = Module::parse(&ctx, include_str!("mlir/batch/square.mlir"))
        .expect("failed to parse batch test module");

    let pm = PassManager::new(&ctx);
    pm.add_pass(pass::create_batch_pass());
    pm.run(&mut module).expect("enzyme-batch pass failed");

    filecheck(&module.as_operation().to_string(), "mlir/batch/square.mlir", None);
}

// Batches the scalar `square` over four inputs, compiles it and runs it: each output is the
// square of its input.
#[test]
fn batch_square_runs() {
    let engine = compile(include_str!("mlir/batch/square_jit.mlir"));

    let square4: extern "C" fn(*const f64, *mut f64) = unsafe { lookup(&engine, "square4") };
    let x = [-2.0, 0.5, 1.0, 3.0];
    let mut out = [0.0; 4];
    square4(x.as_ptr(), out.as_mut_ptr());
    assert_eq!(out, x.map(|x| x * x));
}

// Verifies that all three new pass constructors produce passes that run
// without panicking, even on an empty module.
#[test]
fn batch_and_cleanup_passes_are_runnable() {
    let ctx = create_context();
    let mut module = Module::new(Location::unknown(&ctx));
    let pm = PassManager::new(&ctx);
    pm.add_pass(pass::create_batch_pass());
    pm.add_pass(pass::create_batch_diff_pass());
    pm.add_pass(pass::create_remove_unused_enzyme_ops_pass());
    pm.run(&mut module)
        .expect("batch and cleanup passes should run on an empty module");
}

// Batches a reverse-mode gradient, the score ∇log p of a Gaussian mixture, over 8 points, and checks
// each against the closed form. Ignored because enzyme-batch asserts on scalar constants like
// `arith.constant 0.5 : f64`: its batch rule for `arith.constant` only handles tensor constants
// whose elements are all equal. The assertion aborts this test binary when it runs.
#[test]
#[ignore = "enzyme-batch asserts on scalar arith.constant ops"]
fn batch_of_gradient_runs() {
    let engine = compile(include_str!("mlir/batch/score.mlir"));
    let scores: extern "C" fn(*const f64, *mut f64) = unsafe { lookup(&engine, "scores") };

    let xs = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
    let mut out = [0.0; 8];
    scores(xs.as_ptr(), out.as_mut_ptr());
    // d/dx log(N(x; 2, 1) + N(x; -2, 1)) = (-(x-2) pa - (x+2) pb) / (pa + pb).
    for (x, g) in xs.iter().zip(out) {
        let (pa, pb) = ((-(x - 2.0) * (x - 2.0) / 2.0).exp(), (-(x + 2.0) * (x + 2.0) / 2.0).exp());
        let expected = (-(x - 2.0) * pa - (x + 2.0) * pb) / (pa + pb);
        assert!((g - expected).abs() < 1e-9, "x = {x}: got {g}, expected {expected}");
    }
}
