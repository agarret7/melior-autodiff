use melior::ir::Module;
use melior::pass::bufferization::create_one_shot_bufferize_pass;
use melior::pass::conversion::{
    create_reconcile_unrealized_casts, create_scf_to_control_flow, create_to_llvm,
};
use melior::pass::linalg::{
    create_convert_elementwise_to_linalg_pass, create_convert_linalg_to_loops_pass,
};
use melior::pass::transform::{
    create_canonicalizer_pass, create_cse_pass, create_inliner_pass, create_symbol_dce_pass,
};
use melior::pass::PassManager;
use melior::ExecutionEngine;
use melior_autodiff::{
    create_context, create_lower_enzyme_helpers_pass, enzymeCreateConvertEnzymeToMemRefPass,
    enzymeCreateDifferentiatePass,
};

fn compile(src: &str) -> ExecutionEngine {
    let ctx = create_context();
    let mut module = Module::parse(&ctx, src).expect("parse failed");
    let pm = PassManager::new(&ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });
    pm.add_pass(create_lower_enzyme_helpers_pass());
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateConvertEnzymeToMemRefPass) });
    pm.add_pass(create_inliner_pass());
    pm.add_pass(create_canonicalizer_pass());
    pm.add_pass(create_cse_pass());
    pm.add_pass(create_symbol_dce_pass());
    pm.add_pass(create_convert_elementwise_to_linalg_pass());
    pm.add_pass(create_one_shot_bufferize_pass());
    pm.add_pass(create_convert_linalg_to_loops_pass());
    pm.add_pass(create_scf_to_control_flow());
    pm.add_pass(create_to_llvm());
    pm.add_pass(create_reconcile_unrealized_casts());
    if let Err(e) = pm.run(&mut module) {
        panic!("lowering failed: {e}\n{}", module.as_operation());
    }
    ExecutionEngine::new(&module, 2, &[], false, false)
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[repr(C)]
struct MemRef1 {
    allocated: *mut f64,
    aligned: *mut f64,
    offset: i64,
    sizes: [i64; 1],
    strides: [i64; 1],
}

impl MemRef1 {
    fn new<const N: usize>(data: &mut [f64; N]) -> Self {
        MemRef1 {
            allocated: data.as_mut_ptr(),
            aligned: data.as_mut_ptr(),
            offset: 0,
            sizes: [N as i64],
            strides: [1],
        }
    }
}

// Width-2 forward mode emits enzyme.broadcast of the scalar primal; lowered to tensor.splat.
// d/dx x² = 2x, applied to both tangents at once.
#[test]
fn broadcast_width2_fwddiff() {
    let engine = compile(include_str!("mlir/lower/broadcast_width2.mlir"));
    type Jvp2 = unsafe extern "C" fn(f64, f64, f64, *mut MemRef1);
    let jvp2: Jvp2 = unsafe { std::mem::transmute(engine.lookup("_mlir_ciface_jvp2")) };

    let mut out = [0.0; 2];
    unsafe { jvp2(3.0, 1.0, -2.0, &mut MemRef1::new(&mut out)) };
    assert!(
        approx(out[0], 6.0) && approx(out[1], -12.0),
        "got {out:?}, expected [6, -12]"
    );
}

// Reverse mode through a memref.alloca emits enzyme.fill_zero to zero the shadow buffer.
// f(x) = x² staged through memory,  f'(x) = 2x.
#[test]
fn fill_zero_reverse_alloca() {
    let engine = compile(include_str!("mlir/lower/square_mem.mlir"));
    type Grad = unsafe extern "C" fn(f64, f64) -> f64;
    let grad: Grad = unsafe { std::mem::transmute(engine.lookup("_mlir_ciface_dsquare_mem")) };
    for x in [-2.0, 0.5, 3.0] {
        let g = unsafe { grad(x, 1.0) };
        assert!(approx(g, 2.0 * x), "x={x}: got {g}, expected {}", 2.0 * x);
    }
}
