use melior::ir::{Location, Module};
use melior::pass::bufferization::create_one_shot_bufferize_pass;
use melior::pass::conversion::{
    create_reconcile_unrealized_casts, create_scf_to_control_flow, create_to_llvm,
};
use melior::pass::linalg::{
    create_convert_elementwise_to_linalg_pass, create_convert_linalg_to_loops_pass,
};
use melior::pass::transform::{create_canonicalizer_pass, create_inliner_pass};
use melior::pass::PassManager;
use melior::ExecutionEngine;
use melior_autodiff::{pass, utility::create_context};

mod filecheck;
use filecheck::filecheck;

// Runs enzyme-batch on a parsed module and checks that the batch op is lowered
// to a call to a generated @batched_* function.
#[test]
fn batch_pass_lowers_batch_op() {
    let ctx = create_context();
    let mut module = Module::parse(&ctx, include_str!("mlir/batch/square.mlir"))
        .expect("failed to parse batch test module");

    let pm = PassManager::new(&ctx);
    pm.add_pass(pass::create_batch_pass());
    pm.run(&mut module).expect("enzyme-batch pass failed");

    filecheck(&module.as_operation().to_string(), "mlir/batch/square.mlir", None);
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

// Batches the scalar `square` over four inputs, compiles it and runs it: each output is the
// square of its input.
#[test]
fn batch_square_runs() {
    let ctx = create_context();
    let mut module = Module::parse(&ctx, include_str!("mlir/batch/square_jit.mlir"))
        .expect("failed to parse batch test module");
    let pm = PassManager::new(&ctx);
    pm.add_pass(pass::create_batch_pass());
    pm.add_pass(create_inliner_pass());
    pm.add_pass(create_canonicalizer_pass());
    pm.add_pass(create_convert_elementwise_to_linalg_pass());
    pm.add_pass(create_one_shot_bufferize_pass());
    pm.add_pass(create_convert_linalg_to_loops_pass());
    pm.add_pass(create_scf_to_control_flow());
    pm.add_pass(create_to_llvm());
    pm.add_pass(create_reconcile_unrealized_casts());
    if let Err(e) = pm.run(&mut module) {
        panic!("lowering failed: {e}\n{}", module.as_operation());
    }
    let engine = ExecutionEngine::new(&module, 2, &[], false, false);

    type Square4 = unsafe extern "C" fn(*mut MemRef1, *mut MemRef1);
    let square4: Square4 = unsafe { std::mem::transmute(engine.lookup("_mlir_ciface_square4")) };
    let mut x = [-2.0, 0.5, 1.0, 3.0];
    let mut out = [0.0; 4];
    unsafe { square4(&mut MemRef1::new(&mut x), &mut MemRef1::new(&mut out)) };
    assert_eq!(out, x.map(|x| x * x));
}

// Verifies that all three new pass constructors produce passes that run
// without panicking, even on an empty module.
#[test]
fn batch_and_cleanup_passes_are_runnable() {
    let ctx = create_context();
    let mut module = Module::new(Location::unknown(&ctx));
    let pm = PassManager::new(&ctx);
    pm.add_pass(pass::create_batch_pass());
    pm.add_pass(pass::create_batch_diff_pass());
    pm.add_pass(pass::create_remove_unused_enzyme_ops_pass());
    pm.run(&mut module)
        .expect("batch and cleanup passes should run on an empty module");
}
