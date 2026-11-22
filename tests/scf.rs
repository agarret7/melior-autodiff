mod common;
use common::{approx, compile, lookup};

const TRACE4_MODULE: &str = include_str!("mlir/scf/trace4.mlir");

// f(x) = max(0, x).  JVP: dx if x > 0, else 0.
#[test]
fn grad_relu() {
    let engine = compile(include_str!("mlir/scf/relu.mlir"));

    let jvp: extern "C" fn(f64, f64) -> f64 = unsafe { lookup(&engine, "jvp_relu") };

    // x > 0: tangent passes through
    let r = jvp(2.0, 1.0);
    assert!(approx(r, 1.0), "relu'(2.0)*1.0 = {r}, expected 1.0");

    // x < 0: tangent is zero
    let r = jvp(-1.0, 1.0);
    assert!(approx(r, 0.0), "relu'(-1.0)*1.0 = {r}, expected 0.0");

    // non-unit tangent
    let r = jvp(3.0, 5.0);
    assert!(approx(r, 5.0), "relu'(3.0)*5.0 = {r}, expected 5.0");
}

// f = 9 steps of s <- cos(s^2) from s = x, so f'(x) = prod over the steps of -2 s sin(s^2).
fn grad_iterate_cos(x: f64) -> f64 {
    let (mut s, mut grad) = (x, 1.0);
    for _ in 0..9 {
        grad *= -2.0 * s * f64::sin(s * s);
        s = f64::cos(s * s);
    }
    grad
}

// Reverse mode through a loop with binomial checkpointing, which recomputes steps from 3 snapshots
// instead of caching all 9, against the same loop without checkpointing and the chain rule.
#[test]
fn grad_binomial_checkpointing() {
    let engine = compile(include_str!("mlir/scf/iterate_cos.mlir"));
    type GradFn = extern "C" fn(f64, f64) -> f64;
    let checkpointed: GradFn = unsafe { lookup(&engine, "grad_iterate_checkpointed") };
    let cached: GradFn = unsafe { lookup(&engine, "grad_iterate") };

    for x in [0.3, 0.9, -0.7] {
        let expected = grad_iterate_cos(x);
        let (r, plain) = (checkpointed(x, 1.0), cached(x, 1.0));
        assert!(
            approx(r, expected),
            "checkpointed f'({x}) = {r}, expected {expected}"
        );
        assert!(
            approx(plain, expected),
            "f'({x}) = {plain}, expected {expected}"
        );
    }
}

// A 4x4 matrix, passed to memref<4x4xf64> as a pointer to its row-major data.
type Matrix = [[f64; 4]; 4];
type TraceFn = extern "C" fn(*const Matrix) -> f64;
type JvpTraceFn = extern "C" fn(*const Matrix, *const Matrix) -> f64;

#[test]
fn primal_trace4() {
    let engine = compile(TRACE4_MODULE);
    let f: TraceFn = unsafe { lookup(&engine, "trace4") };
    let eye: Matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let out = f(&eye);
    assert!(approx(out, 4.0), "trace(I) = {out}, expected 4");
}

// trace is linear: fwddiff(@trace4, A, dA) = trace(dA).
#[test]
fn grad_trace4() {
    let engine = compile(TRACE4_MODULE);
    let jvp: JvpTraceFn = unsafe { lookup(&engine, "jvp_trace4") };

    let a: Matrix = [
        [2.0, 1.0, 0.0, 0.0],
        [0.0, 3.0, 1.0, 0.0],
        [0.0, 0.0, 4.0, 1.0],
        [0.0, 0.0, 0.0, 5.0],
    ];

    let eye: Matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let r = jvp(&a, &eye);
    assert!(approx(r, 4.0), "jvp with dA=I: {r}, expected 4");

    let mut e01: [[f64; 4]; 4] = [[0.0; 4]; 4];
    e01[0][1] = 1.0;
    let r = jvp(&a, &e01);
    assert!(approx(r, 0.0), "jvp with dA=e_01: {r}, expected 0");

    let mut e22: [[f64; 4]; 4] = [[0.0; 4]; 4];
    e22[2][2] = 1.0;
    let r = jvp(&a, &e22);
    assert!(approx(r, 1.0), "jvp with dA=e_22: {r}, expected 1");
}
