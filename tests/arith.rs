mod common;
use common::{approx, compile, lookup};

type Derivative1 = extern "C" fn(f64, f64) -> f64;

fn finite_diff(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-5;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

// f(x) = x²,  f'(x) = 2x  — end-to-end reverse-mode smoke test
#[test]
fn grad_square() {
    let engine = compile(include_str!("mlir/arith/square.mlir"));
    let dsquare: Derivative1 = unsafe { lookup(&engine, "dsquare") };
    let result = dsquare(3.0, 1.0);
    assert!(approx(result, 6.0), "dsquare(3, 1) = {result}, expected 6");
}

// f(x) = x³,  f'(x) = 3x²
#[test]
fn grad_cube() {
    let engine = compile(include_str!("mlir/arith/cube.mlir"));
    let dcube: Derivative1 = unsafe { lookup(&engine, "dcube") };
    for x in [-2.0, -1.0, 0.5, 1.0, 3.0] {
        let enzyme = dcube(x, 1.0);
        let fd = finite_diff(|t| t * t * t, x);
        assert!(approx(enzyme, fd), "x={x}: enzyme={enzyme}, fd={fd}");
    }
}

// f(x) = x² - 3x + 1,  f'(x) = 2x - 3
#[test]
fn grad_polynomial() {
    let engine = compile(include_str!("mlir/arith/poly.mlir"));
    let dpoly: Derivative1 = unsafe { lookup(&engine, "dpoly") };
    for x in [-2.0, 0.0, 1.5, 3.0] {
        let enzyme = dpoly(x, 1.0);
        let fd = finite_diff(|t| t * t - 3.0 * t + 1.0, x);
        assert!(approx(enzyme, fd), "x={x}: enzyme={enzyme}, fd={fd}");
    }
}

// f(x, y) = x * y,  ∂f/∂x = y.  y is enzyme_const, so it takes no shadow operand.
#[test]
fn grad_product_wrt_x() {
    let engine = compile(include_str!("mlir/arith/prod.mlir"));
    let dprod_dx: extern "C" fn(f64, f64, f64) -> f64 = unsafe { lookup(&engine, "dprod_dx") };
    for (x, y, dr) in [(1.0, 2.0, 1.0), (-3.0, 4.0, 1.0), (0.5, 0.5, 2.0)] {
        let out = dprod_dx(x, y, dr);
        assert!(
            approx(out, y * dr),
            "∂(x*y)/∂x at ({x},{y}) seed {dr}: enzyme={out}, expected={}",
            y * dr
        );
    }
}
