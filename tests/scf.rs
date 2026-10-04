use melior::pass::conversion::{
    create_reconcile_unrealized_casts, create_scf_to_control_flow, create_to_llvm,
};
use melior::pass::transform::{
    create_canonicalizer_pass, create_cse_pass, create_inliner_pass, create_symbol_dce_pass,
};
use melior::pass::PassManager;
use melior::ExecutionEngine;
use melior::{ir::Module, Context};
use melior_autodiff::{
    enzymeCreateConvertEnzymeToMemRefPass, enzymeCreateDifferentiatePass, Activity,
};

mod common;
use common::{gen_fwddiff, setup_context};

fn compile<'a>(ctx: &'a Context, src: &str) -> (Module<'a>, ExecutionEngine) {
    let mut module = Module::parse(ctx, src).expect("parse failed");
    derive_wrappers(ctx, &module, src);
    let pm = PassManager::new(ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateConvertEnzymeToMemRefPass) });
    pm.add_pass(create_inliner_pass());
    pm.add_pass(create_canonicalizer_pass());
    pm.add_pass(create_cse_pass());
    pm.add_pass(create_symbol_dce_pass());
    pm.add_pass(create_scf_to_control_flow());
    pm.add_pass(create_to_llvm());
    pm.add_pass(create_reconcile_unrealized_casts());
    pm.run(&mut module).expect("lowering failed");
    let engine = ExecutionEngine::new(&module, 2, &[], false, false);
    (module, engine)
}

fn derive_wrappers(ctx: &Context, module: &Module<'_>, src: &str) {
    if src.contains("func.func @trace4") {
        gen_fwddiff(ctx, module, "jvp_trace4", "trace4",
            &["memref<4x4xf64>", "memref<4x4xf64>"], &["f64"],
            &[Activity::Dup], &[Activity::DupNoNeed], true);
    }
    if src.contains("func.func @relu") {
        gen_fwddiff(ctx, module, "jvp_relu", "relu",
            &["f64", "f64"], &["f64"],
            &[Activity::Dup], &[Activity::DupNoNeed], true);
    }
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[repr(C)]
struct MemRef<const R: usize> {
    allocated: *mut f64,
    aligned: *mut f64,
    offset: i64,
    sizes: [i64; R],
    strides: [i64; R],
}

impl MemRef<2> {
    fn from_2d<const M: usize, const N: usize>(data: &mut [[f64; N]; M]) -> Self {
        MemRef {
            allocated: data.as_mut_ptr() as *mut f64,
            aligned: data.as_mut_ptr() as *mut f64,
            offset: 0,
            sizes: [M as i64, N as i64],
            strides: [N as i64, 1],
        }
    }
}

const TRACE4_MODULE: &str = r#"
module {
  func.func @trace4(%A: memref<4x4xf64>) -> f64
      attributes { llvm.emit_c_interface } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %z  = arith.constant 0.0 : f64
    %r  = scf.for %i = %c0 to %c4 step %c1 iter_args(%s = %z) -> f64 {
      %a  = memref.load %A[%i, %i] : memref<4x4xf64>
      %s2 = arith.addf %s, %a : f64
      scf.yield %s2 : f64
    }
    return %r : f64
  }
}
"#;

// f(x) = max(0, x).  JVP: dx if x > 0, else 0.
#[test]
fn grad_relu() {
    let ctx = setup_context();
    let (_, engine) = compile(&ctx, r#"
module {
  func.func @relu(%x: f64) -> f64 attributes { llvm.emit_c_interface } {
    %zero = arith.constant 0.0 : f64
    %cond = arith.cmpf ogt, %x, %zero : f64
    %result = scf.if %cond -> f64 {
      scf.yield %x : f64
    } else {
      scf.yield %zero : f64
    }
    return %result : f64
  }
}
"#);

    type JvpFn = unsafe extern "C" fn(f64, f64) -> f64;
    let jvp: JvpFn = unsafe { std::mem::transmute(engine.lookup("_mlir_ciface_jvp_relu")) };

    // x > 0: tangent passes through
    let r = unsafe { jvp(2.0, 1.0) };
    assert!(approx(r, 1.0), "relu'(2.0)*1.0 = {r}, expected 1.0");

    // x < 0: tangent is zero
    let r = unsafe { jvp(-1.0, 1.0) };
    assert!(approx(r, 0.0), "relu'(-1.0)*1.0 = {r}, expected 0.0");

    // non-unit tangent
    let r = unsafe { jvp(3.0, 5.0) };
    assert!(approx(r, 5.0), "relu'(3.0)*5.0 = {r}, expected 5.0");
}

type TraceFn = unsafe extern "C" fn(*mut MemRef<2>) -> f64;
type JvpTraceFn = unsafe extern "C" fn(*mut MemRef<2>, *mut MemRef<2>) -> f64;

#[test]
fn primal_trace4() {
    let ctx = setup_context();
    let (_, engine) = compile(&ctx, TRACE4_MODULE);
    let f: TraceFn = unsafe { std::mem::transmute(engine.lookup("_mlir_ciface_trace4")) };
    let mut eye: [[f64; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let out = unsafe { f(&mut MemRef::<2>::from_2d(&mut eye)) };
    assert!(approx(out, 4.0), "trace(I) = {out}, expected 4");
}

// trace is linear: fwddiff(@trace4, A, dA) = trace(dA).
#[test]
fn grad_trace4() {
    let ctx = setup_context();
    let (_, engine) = compile(&ctx, TRACE4_MODULE);
    let jvp: JvpTraceFn =
        unsafe { std::mem::transmute(engine.lookup("_mlir_ciface_jvp_trace4")) };

    let mut a: [[f64; 4]; 4] = [
        [2.0, 1.0, 0.0, 0.0],
        [0.0, 3.0, 1.0, 0.0],
        [0.0, 0.0, 4.0, 1.0],
        [0.0, 0.0, 0.0, 5.0],
    ];

    let mut eye: [[f64; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let r = unsafe { jvp(&mut MemRef::<2>::from_2d(&mut a), &mut MemRef::<2>::from_2d(&mut eye)) };
    assert!(approx(r, 4.0), "jvp with dA=I: {r}, expected 4");

    let mut e01: [[f64; 4]; 4] = [[0.0; 4]; 4];
    e01[0][1] = 1.0;
    let r = unsafe { jvp(&mut MemRef::<2>::from_2d(&mut a), &mut MemRef::<2>::from_2d(&mut e01)) };
    assert!(approx(r, 0.0), "jvp with dA=e_01: {r}, expected 0");

    let mut e22: [[f64; 4]; 4] = [[0.0; 4]; 4];
    e22[2][2] = 1.0;
    let r = unsafe { jvp(&mut MemRef::<2>::from_2d(&mut a), &mut MemRef::<2>::from_2d(&mut e22)) };
    assert!(approx(r, 1.0), "jvp with dA=e_22: {r}, expected 1");
}
