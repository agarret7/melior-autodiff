//! An example for each variant of Enzyme's supported activities.
//!
//! `cargo run --example activities`
//!
//! Each function is replaced by its derivative with `enzyme-wrap` (`add_differentiate_wrapper_pass`
//! with no output name), which writes the whole derivative from just a name and activities. The
//! example prints the signature Enzyme generates for each, then calls it.
//!
//! | activity | on an argument | on a result |
//! |---|---|---|
//! | `Active` (reverse) | passed; its gradient is returned | adds `dr`, its incoming gradient |
//! | `Const` | passed, not differentiated | — |
//! | `Dup` (reverse) | passed with a shadow the gradient is added into | — |
//! | `Dup` (forward) | passed with its tangent | returns the value and its tangent |
//! | `DupNoNeed` (forward) | — | returns only the tangent |

mod common;

use melior::{ir::Module, pass::PassManager};
use melior_autodiff::{
    pass::{self, DerivativeMode},
    utility::create_context,
    Activity,
};

const SOURCE: &str = r#"
module {
  // x²
  func.func @active(%x: f64) -> f64 {
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
  // c·x²
  func.func @constant(%x: f64, %c: f64) -> f64 {
    %x2 = arith.mulf %x, %x : f64
    %r = arith.mulf %c, %x2 : f64
    return %r : f64
  }
  // Σ xᵢ²
  func.func @shadow(%x: memref<4xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f64
    %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> f64 {
      %xi = memref.load %x[%i] : memref<4xf64>
      %sq = arith.mulf %xi, %xi : f64
      %next = arith.addf %acc, %sq : f64
      scf.yield %next : f64
    }
    return %sum : f64
  }
  // x³, in forward mode with only its tangent returned
  func.func @tangent(%x: f64) -> f64 {
    %x2 = arith.mulf %x, %x : f64
    %r = arith.mulf %x2, %x : f64
    return %r : f64
  }
  // x³, in forward mode with its value and tangent returned
  func.func @dual(%x: f64) -> f64 {
    %x2 = arith.mulf %x, %x : f64
    %r = arith.mulf %x2, %x : f64
    return %r : f64
  }
}
"#;

/// A value and its tangent, as two `f64` results come back by value under the C ABI.
#[repr(C)]
#[derive(Debug)]
struct Dual {
    value: f64,
    tangent: f64,
}

fn main() {
    use Activity::*;
    use DerivativeMode::*;

    let context = create_context();
    let mut module = Module::parse(&context, SOURCE).expect("invalid MLIR");
    let pm = PassManager::new(&context);
    for (name, arguments, result, mode) in [
        ("active", &[Active][..], Active, ReverseModeCombined),
        ("constant", &[Active, Const][..], Active, ReverseModeCombined),
        ("shadow", &[Dup][..], Active, ReverseModeCombined),
        ("tangent", &[Dup][..], DupNoNeed, ForwardMode),
        ("dual", &[Dup][..], Dup, ForwardMode),
    ] {
        pass::add_differentiate_wrapper_pass(&pm, name, None, arguments, &[result], mode)
            .expect("invalid enzyme-wrap options");
    }
    pass::add_enzyme_lowering(&pm);
    pm.run(&mut module).expect("differentiation failed");

    let text = module.as_operation().to_string();
    for name in ["active", "constant", "shadow", "tangent", "dual"] {
        let line = text.lines().find(|line| line.contains(&format!("@{name}("))).unwrap();
        println!("{}", line.trim().trim_end_matches(" {"));
    }
    println!();

    let engine = common::jit(&context, &mut module);

    // active(x, dr) -> d(x²)/dx · dr = 2x · dr
    let active: extern "C" fn(f64, f64) -> f64 = unsafe { common::lookup(&engine, "active") };
    println!("active(3, dr = 1)      = {}", active(3.0, 1.0));
    assert_eq!(active(3.0, 1.0), 6.0);

    // constant(x, c, dr) -> ∂(c·x²)/∂x · dr = 2cx · dr; c is read, not differentiated
    let constant: extern "C" fn(f64, f64, f64) -> f64 =
        unsafe { common::lookup(&engine, "constant") };
    println!("constant(3, c = 5, 1)  = {}", constant(3.0, 5.0, 1.0));
    assert_eq!(constant(3.0, 5.0, 1.0), 30.0);

    // shadow(x, dx, dr): adds ∂(Σ xᵢ²)/∂x · dr = 2x · dr into dx
    let shadow: extern "C" fn(*const f64, *mut f64, f64) =
        unsafe { common::lookup(&engine, "shadow") };
    let x = [1.0, -2.0, 0.5, 3.0];
    let mut dx = [0.0; 4];
    shadow(x.as_ptr(), dx.as_mut_ptr(), 1.0);
    println!("shadow([1, -2, 0.5, 3]) adds {dx:?} into dx");
    assert_eq!(dx, [2.0, -4.0, 1.0, 6.0]);

    // tangent(x, dx) -> d(x³)/dx · dx = 3x² · dx
    let tangent: extern "C" fn(f64, f64) -> f64 = unsafe { common::lookup(&engine, "tangent") };
    println!("tangent(2, dx = 1)     = {}", tangent(2.0, 1.0));
    assert_eq!(tangent(2.0, 1.0), 12.0);

    // dual(x, dx) -> (x³, 3x² · dx)
    let dual: extern "C" fn(f64, f64) -> Dual = unsafe { common::lookup(&engine, "dual") };
    let result = dual(2.0, 1.0);
    println!("dual(2, dx = 1)        = {result:?}");
    assert_eq!((result.value, result.tangent), (8.0, 12.0));
}
