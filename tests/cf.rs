// Differentiating unstructured control flow: `cf.br` and `cf.cond_br` between blocks, the form
// `scf` takes after `convert-scf-to-cf`, or that a frontend emitting a CFG produces directly.
//
// Reverse mode is ignored for now: Enzyme's reverse pass routes each block argument's adjoint to a
// predecessor by comparing against the argument's number instead of the predecessor's index (a
// shadowed `idx` in `EnzymeLogicReverse.cpp`), so every predecessor but the first gets a zero
// gradient. The two tests pass with the fix.

mod common;
use common::{approx, compile, lookup};

// f(x) = x² if x > 0, else -x: two branches joined by a block argument.
// f'(x) = 2x if x > 0, else -1.
const PIECEWISE: &str = include_str!("mlir/cf/piecewise.mlir");

fn piecewise_derivative(x: f64) -> f64 {
    if x > 0.0 {
        2.0 * x
    } else {
        -1.0
    }
}

#[test]
fn fwddiff_through_branches() {
    let engine = compile(PIECEWISE);
    let fwd: extern "C" fn(f64, f64) -> f64 = unsafe { lookup(&engine, "fwd_piecewise") };
    for x in [-2.0, -0.5, 0.5, 3.0] {
        let got = fwd(x, 1.0);
        assert!(approx(got, piecewise_derivative(x)), "x={x}: got {got}");
    }
}

#[test]
#[ignore = "Enzyme routes block-argument adjoints by argument number, not predecessor"]
fn autodiff_through_branches() {
    let engine = compile(PIECEWISE);
    let rev: extern "C" fn(f64, f64) -> f64 = unsafe { lookup(&engine, "rev_piecewise") };
    for x in [-2.0, -0.5, 0.5, 3.0] {
        let got = rev(x, 1.0);
        assert!(approx(got, piecewise_derivative(x)), "x={x}: got {got}");
    }
}

// x^n by repeated multiplication, as a loop of blocks with a back edge, so the trip count n is only
// known at runtime: reverse mode has to cache the accumulator on each iteration to run it backwards.
// d/dx x^n = n x^(n-1).
const POWER: &str = include_str!("mlir/cf/power.mlir");

fn power_derivative(x: f64, n: i64) -> f64 {
    if n == 0 {
        0.0
    } else {
        n as f64 * x.powi(n as i32 - 1)
    }
}

#[test]
fn fwddiff_through_a_loop() {
    let engine = compile(POWER);
    let fwd: extern "C" fn(f64, f64, i64) -> f64 = unsafe { lookup(&engine, "fwd_power") };
    for n in 0..6 {
        for x in [-1.5, 0.5, 2.0] {
            let got = fwd(x, 1.0, n);
            assert!(
                approx(got, power_derivative(x, n)),
                "x={x}, n={n}: got {got}"
            );
        }
    }
}

#[test]
#[ignore = "Enzyme routes block-argument adjoints by argument number, not predecessor"]
fn autodiff_through_a_loop() {
    let engine = compile(POWER);
    let rev: extern "C" fn(f64, i64, f64) -> f64 = unsafe { lookup(&engine, "rev_power") };
    for n in 0..6 {
        for x in [-1.5, 0.5, 2.0] {
            let got = rev(x, n, 1.0);
            assert!(
                approx(got, power_derivative(x, n)),
                "x={x}, n={n}: got {got}"
            );
        }
    }
}
