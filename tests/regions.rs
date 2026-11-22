mod common;
use common::{approx, compile, lookup};

const SCALED_MODULE: &str = include_str!("mlir/regions/scaled.mlir");

// f(x) = c x^2 inside `enzyme.autodiff_region`, capturing c: f'(x) = 2 c x.
#[test]
fn grad_autodiff_region() {
    let engine = compile(SCALED_MODULE);
    let grad: extern "C" fn(f64, f64) -> f64 = unsafe { lookup(&engine, "grad_scaled_square") };

    for (c, x) in [(3.0, 2.0), (-0.5, 4.0), (2.0, -1.5)] {
        let r = grad(c, x);
        assert!(
            approx(r, 2.0 * c * x),
            "f'({x}) with c = {c}: {r}, expected {}",
            2.0 * c * x
        );
    }
}

// f(x) = c sin(x) inside `enzyme.fwddiff_region`, capturing c: f'(x) dx = c cos(x) dx.
#[test]
fn jvp_fwddiff_region() {
    let engine = compile(SCALED_MODULE);
    let jvp: extern "C" fn(f64, f64, f64) -> f64 = unsafe { lookup(&engine, "jvp_scaled_sin") };

    for (c, x, dx) in [(3.0, 0.5, 1.0), (-2.0, 1.2, 0.5), (0.5, -2.0, 3.0)] {
        let expected = c * f64::cos(x) * dx;
        let r = jvp(c, x, dx);
        assert!(
            approx(r, expected),
            "jvp at x = {x}, dx = {dx}, c = {c}: {r}, expected {expected}"
        );
    }
}
