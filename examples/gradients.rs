//! Reverse-mode gradients built from Rust with this crate's `autodiff`: of a scalar function, and
//! of a function of an array.
//!
//! `cargo run --example gradients`
//!
//! The functions to differentiate are parsed from MLIR; the functions that compute their
//! gradients are built with melior and `autodiff`, the way a frontend generating IR would. Each
//! writes its gradient into buffers, so from Rust it's an ordinary C function.

mod common;

use melior::{
    dialect::{arith, memref},
    ir::{
        attribute::FloatAttribute, r#type::MemRefType, Block, BlockLike, Location, Module, Type,
        Value,
    },
    pass::PassManager,
    Context,
};
use melior_autodiff::{autodiff, pass, utility::create_context, Activity};

const SOURCE: &str = r#"
module {
  // f(x, y) = x·y + sin(x)
  func.func @f(%x: f64, %y: f64) -> f64 {
    %xy = arith.mulf %x, %y : f64
    %s = math.sin %x : f64
    %r = arith.addf %xy, %s : f64
    return %r : f64
  }

  // g(x) = Σ xᵢ² · wᵢ
  func.func @g(%x: memref<4xf64>, %w: memref<4xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f64
    %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> f64 {
      %xi = memref.load %x[%i] : memref<4xf64>
      %wi = memref.load %w[%i] : memref<4xf64>
      %sq = arith.mulf %xi, %xi : f64
      %term = arith.mulf %sq, %wi : f64
      %next = arith.addf %acc, %term : f64
      scf.yield %next : f64
    }
    return %sum : f64
  }
}
"#;

/// The constant 1.0: the incoming gradient of a scalar result, for the plain gradient.
fn one<'c, 'a>(context: &'c Context, block: &'a Block<'c>) -> Value<'c, 'a> {
    let f64_type = Type::float64(context);
    let constant = arith::constant(
        context,
        FloatAttribute::new(context, f64_type, 1.0).into(),
        Location::unknown(context),
    );
    block.append_operation(constant).result(0).unwrap().into()
}

fn main() {
    let context = create_context();
    let location = Location::unknown(&context);
    let mut module = Module::parse(&context, SOURCE).expect("invalid MLIR");
    let f64_type = Type::float64(&context);
    let scalar = MemRefType::new(f64_type, &[], None, None).into();
    let array = MemRefType::new(f64_type, &[4], None, None).into();

    // func.func @grad_f(%x: f64, %y: f64, %dx: memref<f64>, %dy: memref<f64>) {
    //   %one = arith.constant 1.0 : f64
    //   %gx, %gy = enzyme.autodiff @f(%x, %y, %one) <{
    //     activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    //     ret_activity = [#enzyme<activity enzyme_activenoneed>]
    //   }> : (f64, f64, f64) -> (f64, f64)
    //   memref.store %gx, %dx[] : memref<f64>
    //   memref.store %gy, %dy[] : memref<f64>
    //   return
    // }
    common::add_function(
        &context,
        &module,
        "grad_f",
        &[f64_type, f64_type, scalar, scalar],
        |block| {
            let [x, y, dx, dy] = [0, 1, 2, 3].map(|i| block.argument(i).unwrap().into());
            let gradient = block.append_operation(autodiff(
                &context,
                "f",
                &[x, y, one(&context, block)],
                &[f64_type, f64_type],
                &[Activity::Active, Activity::Active],
                &[Activity::ActiveNoNeed],
                1,     // width: one derivative at a time
                false, // strong_zero: Enzyme's default
                location,
            ));
            for (index, out) in [dx, dy].into_iter().enumerate() {
                let value = gradient.result(index).unwrap().into();
                block.append_operation(memref::store(value, out, &[], location));
            }
        },
    );

    // func.func @grad_g(%x: memref<4xf64>, %dx: memref<4xf64>, %w: memref<4xf64>) {
    //   %one = arith.constant 1.0 : f64
    //   enzyme.autodiff @g(%x, %dx, %w, %one) <{
    //     activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>],
    //     ret_activity = [#enzyme<activity enzyme_activenoneed>]
    //   }> : (memref<4xf64>, memref<4xf64>, memref<4xf64>, f64) -> ()
    //   return
    // }
    common::add_function(
        &context,
        &module,
        "grad_g",
        &[array, array, array],
        |block| {
            let [x, dx, w] = [0, 1, 2].map(|i| block.argument(i).unwrap().into());
            block.append_operation(autodiff(
                &context,
                "g",
                &[x, dx, w, one(&context, block)],
                &[],
                &[Activity::Dup, Activity::Const],
                &[Activity::ActiveNoNeed],
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

    // grad_f(x, y, dx, dy)
    //   x, y:   where to differentiate; both `Active`
    //   dx, dy: overwritten with ∂f/∂x and ∂f/∂y
    let grad_f: extern "C" fn(f64, f64, *mut f64, *mut f64) =
        unsafe { common::lookup(&engine, "grad_f") };
    let (x, y) = (0.5, 2.0);
    let (mut dx, mut dy) = (0.0, 0.0);
    grad_f(x, y, &mut dx, &mut dy);
    println!("∇f({x}, {y}) = ({dx:.6}, {dy:.6})");
    // ∇f = (y + cos x, x)
    assert!((dx - (y + x.cos())).abs() < 1e-12 && (dy - x).abs() < 1e-12);

    // grad_g(x, dx, w)
    //   x:  where to differentiate, 4 values; `Dup`, so it comes with a shadow
    //   dx: x's shadow, 4 values; ∂g/∂x is *added* into it, so zero it first
    //   w:  4 weights; `Const`, so they're only read
    let grad_g: extern "C" fn(*const f64, *mut f64, *const f64) =
        unsafe { common::lookup(&engine, "grad_g") };
    let x = [1.0, -2.0, 0.5, 3.0];
    let w = [1.0, 0.5, 2.0, -1.0];
    let mut dx = [0.0; 4];
    grad_g(x.as_ptr(), dx.as_mut_ptr(), w.as_ptr());
    println!("∇g(x) = {dx:?}");
    // ∂g/∂xᵢ = 2 xᵢ wᵢ
    let expected: Vec<f64> = x.iter().zip(&w).map(|(x, w)| 2.0 * x * w).collect();
    assert_eq!(dx.to_vec(), expected);
    // A second call without zeroing adds another gradient on top.
    grad_g(x.as_ptr(), dx.as_mut_ptr(), w.as_ptr());
    println!("after a second call without zeroing: {dx:?}");
    assert_eq!(
        dx.to_vec(),
        expected.iter().map(|g| 2.0 * g).collect::<Vec<_>>()
    );
}
