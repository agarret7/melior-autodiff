//! Reverse mode on a vector-valued function: the vector-Jacobian product uᵀJ of
//! f(x) = x · exp(x), elementwise, with the derivative built from Rust through this crate's
//! `autodiff`.
//!
//! `cargo run --example vjp`
//!
//! `jvp` pushes a tangent v forward through f (J·v), while a VJP pulls a cotangent u back from f's
//! output to its input (uᵀJ): the building block of backpropagation. Enzyme's reverse mode works
//! on memrefs, not tensors, so `@f` writes its result into a buffer `y`. Both `x` and `y` are
//! `Dup`: `y`'s shadow carries u in, and `x`'s shadow `dx` is where uᵀJ is added. Reverse mode
//! *consumes* an output's shadow: once u has been propagated, Enzyme zeroes it, so copy u first if
//! you still need it.

mod common;

use melior::{
    dialect::memref,
    ir::{BlockLike, Location, Module, Type, r#type::MemRefType},
    pass::PassManager,
};
use melior_autodiff::{Activity, autodiff, fwddiff, pass, utility::create_context};

const SOURCE: &str = r#"
module {
  // y = x · exp(x), elementwise
  func.func @f(%x: memref<4xf64>, %y: memref<4xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %xi = memref.load %x[%i] : memref<4xf64>
      %e = math.exp %xi : f64
      %yi = arith.mulf %xi, %e : f64
      memref.store %yi, %y[%i] : memref<4xf64>
    }
    return
  }
}
"#;

fn main() {
    let context = create_context();
    let location = Location::unknown(&context);
    let mut module = Module::parse(&context, SOURCE).expect("invalid MLIR");
    let buffer: Type = MemRefType::new(Type::float64(&context), &[4], None, None).into();

    // func.func @vjp_f(%x: memref<4xf64>, %u: memref<4xf64>, %dx: memref<4xf64>) {
    //   %y = memref.alloca() : memref<4xf64>
    //   enzyme.autodiff @f(%x, %dx, %y, %u) <{
    //     activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    //     ret_activity = []
    //   }> : (memref<4xf64>, memref<4xf64>, memref<4xf64>, memref<4xf64>) -> ()
    //   return
    // }
    common::add_function(
        &context,
        &module,
        "vjp_f",
        &[buffer, buffer, buffer],
        |block| {
            let [x, u, dx] = [0, 1, 2].map(|i| block.argument(i).unwrap().into());
            // f's output, which the VJP doesn't return.
            let alloca = memref::alloca(
                &context,
                MemRefType::try_from(buffer).unwrap(),
                &[],
                &[],
                None,
                location,
            );
            let y = block.append_operation(alloca).result(0).unwrap().into();
            block.append_operation(autodiff(
                &context,
                "f",
                &[x, dx, y, u],
                &[],
                &[Activity::Dup, Activity::Dup],
                &[],
                1,     // width: one cotangent at a time
                false, // strong_zero: Enzyme's default
                location,
            ));
        },
    );

    // Forward mode on the same @f, for the check at the end: `y`'s tangent receives J·v.
    //
    // func.func @jvp_f(%x: memref<4xf64>, %v: memref<4xf64>, %jv: memref<4xf64>) {
    //   %y = memref.alloca() : memref<4xf64>
    //   enzyme.fwddiff @f(%x, %v, %y, %jv) <{
    //     activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    //     ret_activity = []
    //   }> : (memref<4xf64>, memref<4xf64>, memref<4xf64>, memref<4xf64>) -> ()
    //   return
    // }
    common::add_function(
        &context,
        &module,
        "jvp_f",
        &[buffer, buffer, buffer],
        |block| {
            let [x, v, jv] = [0, 1, 2].map(|i| block.argument(i).unwrap().into());
            let alloca = memref::alloca(
                &context,
                MemRefType::try_from(buffer).unwrap(),
                &[],
                &[],
                None,
                location,
            );
            let y = block.append_operation(alloca).result(0).unwrap().into();
            block.append_operation(fwddiff(
                &context,
                "f",
                &[x, v, y, jv],
                &[],
                &[Activity::Dup, Activity::Dup],
                &[],
                1,
                false,
                location,
            ));
        },
    );

    let pm = PassManager::new(&context);
    pass::add_enzyme_lowering(&pm);
    pm.run(&mut module).expect("differentiation failed");
    let engine = common::jit(&context, &mut module);

    // vjp_f(x, u, dx)
    //   x:  where to differentiate, 4 values
    //   u:  the cotangent of f's output, 4 values; zeroed by the call
    //   dx: 4 values; uᵀJ is *added* into it, so zero it first
    let vjp_f: extern "C" fn(*const f64, *mut f64, *mut f64) =
        unsafe { common::lookup(&engine, "vjp_f") };

    let x = [0.0, 0.5, -1.0, 2.0];
    let u_in = [1.0, 2.0, 0.0, -0.5];
    let mut u = u_in;
    let mut dx = [0.0; 4];
    vjp_f(x.as_ptr(), u.as_mut_ptr(), dx.as_mut_ptr());

    // f is elementwise, so J is diagonal: (uᵀJ)ᵢ = uᵢ f'(xᵢ) = uᵢ (1 + xᵢ) exp(xᵢ).
    for i in 0..4 {
        let expected = u_in[i] * (1.0 + x[i]) * x[i].exp();
        println!(
            "(uᵀJ)[{i}] = {:.6}   u (1 + x) exp(x) = {expected:.6}",
            dx[i]
        );
        assert!((dx[i] - expected).abs() < 1e-12);
    }
    // The output's shadow was consumed.
    println!("u after the call: {u:?}");
    assert_eq!(u, [0.0; 4]);

    // The dot-product test: for any u and v, u·(J·v) = (uᵀJ)·v. Forward and reverse mode compute the
    // same derivative, so the two sides must agree up to rounding; if they don't, one mode is wrong.
    // It needs no closed form for J, so it works as a check for any f.
    let jvp_f: extern "C" fn(*const f64, *const f64, *mut f64) =
        unsafe { common::lookup(&engine, "jvp_f") };
    let u = [0.3, -1.0, 2.0, 0.5];
    let v = [1.0, 0.25, -0.5, 2.0];
    let mut jv = [0.0; 4];
    jvp_f(x.as_ptr(), v.as_ptr(), jv.as_mut_ptr());
    let mut u_consumed = u;
    let mut ut_j = [0.0; 4];
    vjp_f(x.as_ptr(), u_consumed.as_mut_ptr(), ut_j.as_mut_ptr());

    let dot = |a: &[f64; 4], b: &[f64; 4]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    let (forward, reverse) = (dot(&u, &jv), dot(&ut_j, &v));
    println!("u·(J·v) = {forward:.12}   (uᵀJ)·v = {reverse:.12}");
    assert!((forward - reverse).abs() < 1e-12);
}
