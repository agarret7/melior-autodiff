//! Forward mode on tensors: the Jacobian-vector product of f(x) = x · exp(x), elementwise, with the
//! derivative built from Rust through this crate's `fwddiff`.
//!
//! `cargo run --example jvp`
//!
//! `enzyme.fwddiff` takes each `Dup` argument followed by its *tangent* `v`, and returns the
//! tangent of the result, J·v: how f's output moves when x moves along v. `DupNoNeed` on the
//! result: we want its tangent, not its value. Forward mode works on tensors (reverse mode on
//! tensors doesn't, yet), but functions called from Rust take buffers, so `@jvp_f` converts at its
//! boundary and writes J·v into `out`.

mod common;

use melior::{
    dialect::ods::bufferization::{MaterializeInDestinationOperation, ToTensorOperation},
    ir::{
        r#type::{MemRefType, RankedTensorType},
        Attribute, BlockLike, Location, Module, Type, Value,
    },
    pass::PassManager,
};
use melior_autodiff::{fwddiff, pass, utility::create_context, Activity};

const SOURCE: &str = r#"
module {
  func.func private @f(%x: tensor<4xf64>) -> tensor<4xf64> {
    %e = math.exp %x : tensor<4xf64>
    %y = arith.mulf %x, %e : tensor<4xf64>
    return %y : tensor<4xf64>
  }
}
"#;

fn main() {
    let context = create_context();
    let location = Location::unknown(&context);
    let mut module = Module::parse(&context, SOURCE).expect("invalid MLIR");
    let f64_type = Type::float64(&context);
    let buffer = MemRefType::new(f64_type, &[4], None, None).into();
    let tensor: Type = RankedTensorType::new(&[4], f64_type, None).into();
    let unit = Attribute::unit(&context);

    // func.func @jvp_f(%x: memref<4xf64>, %v: memref<4xf64>, %out: memref<4xf64>) {
    //   %xt = bufferization.to_tensor %x restrict : memref<4xf64> to tensor<4xf64>
    //   %vt = bufferization.to_tensor %v restrict : memref<4xf64> to tensor<4xf64>
    //   %jv = enzyme.fwddiff @f(%xt, %vt) <{
    //     activity = [#enzyme<activity enzyme_dup>],
    //     ret_activity = [#enzyme<activity enzyme_dupnoneed>]
    //   }> : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    //   bufferization.materialize_in_destination %jv in restrict writable %out
    //     : (tensor<4xf64>, memref<4xf64>) -> ()
    //   return
    // }
    common::add_function(&context, &module, "jvp_f", &[buffer, buffer, buffer], |block| {
        let [x, v, out] = [0, 1, 2].map(|i| block.argument(i).unwrap().into());
        let [xt, vt]: [Value; 2] = [x, v].map(|memref| {
            let op = ToTensorOperation::builder(&context, location)
                .result(tensor)
                .buffer(memref)
                .restrict(Some(unit))
                .build();
            block.append_operation(op.into()).result(0).unwrap().into()
        });
        let jv = block.append_operation(fwddiff(
            &context,
            "f",
            &[xt, vt],
            &[tensor],
            &[Activity::Dup],
            &[Activity::DupNoNeed],
            1,     // width: one tangent at a time
            false, // strong_zero: Enzyme's default
            location,
        ));
        let materialize = MaterializeInDestinationOperation::builder(&context, location)
            .source(jv.result(0).unwrap().into())
            .dest(out)
            .restrict(Some(unit))
            .writable(Some(unit))
            .build();
        block.append_operation(materialize.into());
    });

    let pm = PassManager::new(&context);
    pass::add_enzyme_lowering(&pm).expect("invalid lowering pipeline");
    pm.run(&mut module).expect("differentiation failed");
    let engine = common::jit(&context, &mut module);

    // jvp_f(x, v, out)
    //   x:   where to differentiate, 4 values; `Dup`, so it comes with a tangent
    //   v:   x's tangent, 4 values: the direction to move x in
    //   out: 4 values, overwritten with J·v
    let jvp_f: extern "C" fn(*const f64, *const f64, *mut f64) =
        unsafe { common::lookup(&engine, "jvp_f") };

    let x = [0.0, 0.5, -1.0, 2.0];
    let v = [1.0, 2.0, 0.0, -0.5];
    let mut out = [0.0; 4];
    jvp_f(x.as_ptr(), v.as_ptr(), out.as_mut_ptr());

    // f is elementwise, so J is diagonal: (J·v)ᵢ = f'(xᵢ) vᵢ = (1 + xᵢ) exp(xᵢ) vᵢ.
    let f = |x: f64| x * x.exp();
    for i in 0..4 {
        let expected = (1.0 + x[i]) * x[i].exp() * v[i];
        let fd = common::finite_difference(f, x[i]) * v[i];
        println!(
            "(J·v)[{i}] = {:.6}   (1 + x) exp(x) v = {expected:.6}   finite difference = {fd:.6}",
            out[i]
        );
        assert!((out[i] - expected).abs() < 1e-12);
    }
}
