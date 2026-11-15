//! Helpers shared by the tests that JIT-run derivatives.
#![allow(dead_code)]

use melior::{
    ir::Module,
    pass::{bufferization, conversion, linalg, memref, transform, PassManager},
    utility::{parse_pass_pipeline, register_all_passes},
    ExecutionEngine,
};
use melior_autodiff::{pass, utility::create_context};

// Replaces the Enzyme ops, cleans up, and lowers tensors and `linalg` to loops: everything before
// the final conversion to the LLVM dialect.
fn add_lowering_to_loops(pm: &PassManager) {
    pass::add_enzyme_lowering(pm).expect("invalid lowering pipeline");
    pm.add_pass(transform::create_inliner_pass());
    pm.add_pass(transform::create_canonicalizer_pass());
    pm.add_pass(transform::create_cse_pass());
    pm.add_pass(transform::create_symbol_dce_pass());
    pm.add_pass(linalg::create_convert_elementwise_to_linalg_pass());
    pm.add_pass(bufferization::create_one_shot_bufferize_pass());
    pm.add_pass(linalg::create_convert_linalg_to_loops_pass());
    pm.add_pass(conversion::create_scf_to_control_flow());
    // Lowers `memref.subview` (Enzyme indexes its caches through it) and reshapes, emitting affine
    // ops that `lower-affine` lowers.
    pm.add_pass(memref::create_expand_strided_metadata_pass());
    pm.add_pass(conversion::create_lower_affine());
}

/// Parses `src`, lowers it to LLVM and JIT-compiles it, with MLIR's bare-pointer calling
/// convention: a statically shaped memref argument is a plain pointer to its data and scalars are
/// passed by value, so functions are called as ordinary `extern "C" fn`s (see [`lookup`]).
pub fn compile(src: &str) -> ExecutionEngine {
    let ctx = create_context();
    let mut module = Module::parse(&ctx, src).expect("parse failed");
    let pm = PassManager::new(&ctx);
    add_lowering_to_loops(&pm);
    run(&pm, &mut module);

    register_all_passes();
    let to_llvm = PassManager::new(&ctx);
    parse_pass_pipeline(
        to_llvm.as_operation_pass_manager(),
        "builtin.module(convert-math-to-llvm, convert-arith-to-llvm, finalize-memref-to-llvm, \
         convert-cf-to-llvm, convert-func-to-llvm{use-bare-ptr-memref-call-conv=1}, \
         reconcile-unrealized-casts)",
    )
    .expect("invalid pipeline");
    run(&to_llvm, &mut module);
    ExecutionEngine::new(&module, 2, &[], false, false).expect("JIT compilation failed")
}

/// Like [`compile`], with MLIR's default calling convention instead, which dynamically shaped
/// memrefs need: functions marked `llvm.emit_c_interface` get a `_mlir_ciface_` wrapper taking a
/// pointer to each memref's descriptor.
pub fn compile_with_descriptors(src: &str) -> ExecutionEngine {
    let ctx = create_context();
    let mut module = Module::parse(&ctx, src).expect("parse failed");
    let pm = PassManager::new(&ctx);
    add_descriptor_lowering(&pm);
    run(&pm, &mut module);
    ExecutionEngine::new(&module, 2, &[], false, false).expect("JIT compilation failed")
}

/// The passes [`compile_with_descriptors`] runs: to the LLVM dialect, with MLIR's default calling
/// convention.
pub fn add_descriptor_lowering(pm: &PassManager) {
    add_lowering_to_loops(pm);
    pm.add_pass(conversion::create_to_llvm());
    pm.add_pass(conversion::create_reconcile_unrealized_casts());
}

fn run(pm: &PassManager, module: &mut Module) {
    if let Err(e) = pm.run(module) {
        panic!("lowering failed: {e}\n{}", module.as_operation());
    }
}

/// Looks up the JIT-compiled function `name`.
///
/// # Safety
///
/// `F` must be an `extern "C" fn` type matching the function's lowered signature.
pub unsafe fn lookup<F: Copy>(engine: &ExecutionEngine, name: &str) -> F {
    let pointer = engine.lookup(name);
    assert!(!pointer.is_null(), "no function @{name}");
    assert_eq!(size_of::<F>(), size_of::<*mut ()>());
    unsafe { std::mem::transmute_copy(&pointer) }
}

pub fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}
