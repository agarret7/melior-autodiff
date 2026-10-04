use melior::ir::Module;
use melior::pass::conversion::{
    create_reconcile_unrealized_casts, create_scf_to_control_flow, create_to_llvm,
};
use melior::pass::transform::{
    create_canonicalizer_pass, create_cse_pass, create_inliner_pass, create_symbol_dce_pass,
};
use melior::pass::PassManager;
use melior::ExecutionEngine;
use melior_autodiff::{
    create_context, enzymeCreateConvertEnzymeToMemRefPass, enzymeCreateDifferentiatePass,
};

fn compile(src: &str) -> ExecutionEngine {
    let ctx = create_context();
    let mut module = Module::parse(&ctx, src).expect("parse failed");
    let pm = PassManager::new(&ctx);
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
    ExecutionEngine::new(&module, 2, &[], false, false)
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

const TRACE4_MODULE: &str = include_str!("mlir/scf/trace4.mlir");

// f(x) = max(0, x).  JVP: dx if x > 0, else 0.
#[test]
fn grad_relu() {
    let engine = compile(include_str!("mlir/scf/relu.mlir"));

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
    let engine = compile(TRACE4_MODULE);
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
    let engine = compile(TRACE4_MODULE);
    let jvp: JvpTraceFn = unsafe { std::mem::transmute(engine.lookup("_mlir_ciface_jvp_trace4")) };

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
    let r = unsafe {
        jvp(
            &mut MemRef::<2>::from_2d(&mut a),
            &mut MemRef::<2>::from_2d(&mut eye),
        )
    };
    assert!(approx(r, 4.0), "jvp with dA=I: {r}, expected 4");

    let mut e01: [[f64; 4]; 4] = [[0.0; 4]; 4];
    e01[0][1] = 1.0;
    let r = unsafe {
        jvp(
            &mut MemRef::<2>::from_2d(&mut a),
            &mut MemRef::<2>::from_2d(&mut e01),
        )
    };
    assert!(approx(r, 0.0), "jvp with dA=e_01: {r}, expected 0");

    let mut e22: [[f64; 4]; 4] = [[0.0; 4]; 4];
    e22[2][2] = 1.0;
    let r = unsafe {
        jvp(
            &mut MemRef::<2>::from_2d(&mut a),
            &mut MemRef::<2>::from_2d(&mut e22),
        )
    };
    assert!(approx(r, 1.0), "jvp with dA=e_22: {r}, expected 1");
}
