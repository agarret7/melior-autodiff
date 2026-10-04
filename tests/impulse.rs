use melior::ir::Module;
use melior::pass::conversion::{
    create_lower_affine, create_reconcile_unrealized_casts, create_scf_to_control_flow,
    create_to_llvm,
};
use melior::pass::linalg::{
    create_convert_elementwise_to_linalg_pass, create_convert_linalg_to_loops_pass,
};
use melior::pass::memref::create_expand_strided_metadata_pass;
use melior::pass::transform::{
    create_canonicalizer_pass, create_cse_pass, create_inliner_pass, create_symbol_dce_pass,
};
use melior::pass::{OperationPassManager, Pass, PassManager};
use melior::utility::register_all_passes;
use melior::{ExecutionEngine, StringRef};
use melior_autodiff::{
    create_context, create_lower_enzyme_helpers_pass, create_lower_impulse_pass,
    enzymeCreateConvertEnzymeToMemRefPass, enzymeCreateDifferentiatePass,
    enzymeCreateExpandImpulsePass, enzymeCreateOutlineEnzymeFromRegionPass,
    enzymeCreateRemoveUnusedEnzymeOpsPass,
};
use std::sync::Once;

mod filecheck;
use filecheck::filecheck;

fn read(file: &str) -> String {
    std::fs::read_to_string(format!("{}/tests/{file}", env!("CARGO_MANIFEST_DIR")))
        .expect("missing test file")
}

// Expands and differentiates `src`, then FileChecks the result against the lines in tests/<file>.
fn differentiate(src: &str, file: &str, prefix: Option<&str>) {
    let ctx = create_context();
    let mut module = Module::parse(&ctx, src).expect("parse failed");
    let pm = PassManager::new(&ctx);
    unsafe {
        pm.add_pass(Pass::from_raw_fn(enzymeCreateExpandImpulsePass));
        pm.add_pass(Pass::from_raw_fn(enzymeCreateOutlineEnzymeFromRegionPass));
        pm.add_pass(Pass::from_raw_fn(enzymeCreateDifferentiatePass));
    }
    if pm.run(&mut module).is_err() {
        panic!("differentiation failed:\n{}", module.as_operation());
    }
    filecheck(&module.as_operation().to_string(), file, prefix);
}

fn reverse(file: &str) {
    differentiate(&read(file), file, None);
}

#[test]
fn reverse_dynamic_slice() {
    reverse("mlir/impulse/reverse_dynamic_slice.mlir");
}

#[test]
fn reverse_dynamic_update_slice() {
    reverse("mlir/impulse/reverse_dynamic_update_slice.mlir");
}

#[test]
fn reverse_slice() {
    reverse("mlir/impulse/reverse_slice.mlir");
}

#[test]
fn reverse_reshape() {
    reverse("mlir/impulse/reverse_reshape.mlir");
}

// HMC on a Gaussian: expand-impulse emits enzyme.autodiff_region for ∇logpdf, which must
// outline and differentiate through impulse.{dynamic_slice, dynamic_update_slice, slice, reshape}.
#[test]
fn hmc_gaussian_differentiates() {
    differentiate(&HMC_RUN.replace("NN", "10"), HMC_FILE, Some("DIFF"));
}

/// Appends textual pipeline elements to `pm`. melior's `parse_pass_pipeline` uses
/// `mlirParsePassPipeline`, which replaces the passes already in the manager.
fn add_pipeline(pm: OperationPassManager, elements: &str) {
    unsafe extern "C" fn print_error(message: mlir_sys::MlirStringRef, _: *mut std::ffi::c_void) {
        let bytes =
            unsafe { std::slice::from_raw_parts(message.data as *const u8, message.length) };
        eprint!("{}", String::from_utf8_lossy(bytes));
    }
    let result = unsafe {
        mlir_sys::mlirOpPassManagerAddPipeline(
            pm.to_raw(),
            StringRef::new(elements).to_raw(),
            Some(print_error),
            std::ptr::null_mut(),
        )
    };
    assert!(result.value != 0, "failed to parse pipeline: {elements}");
}

// Expand, differentiate, lower Impulse and Enzyme helpers, bufferize, and JIT.
fn compile(src: &str) -> ExecutionEngine {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(register_all_passes);

    let ctx = create_context();
    let mut module = Module::parse(&ctx, src).expect("parse failed");
    let pm = PassManager::new(&ctx);
    unsafe {
        pm.add_pass(Pass::from_raw_fn(enzymeCreateExpandImpulsePass));
        pm.add_pass(Pass::from_raw_fn(enzymeCreateOutlineEnzymeFromRegionPass));
        pm.add_pass(Pass::from_raw_fn(enzymeCreateDifferentiatePass));
    }
    pm.add_pass(create_lower_impulse_pass());
    pm.add_pass(create_canonicalizer_pass());
    pm.add_pass(unsafe { Pass::from_raw_fn(enzymeCreateRemoveUnusedEnzymeOpsPass) });
    pm.add_pass(create_canonicalizer_pass());
    pm.add_pass(create_lower_enzyme_helpers_pass());
    pm.add_pass(unsafe { Pass::from_raw_fn(enzymeCreateConvertEnzymeToMemRefPass) });
    pm.add_pass(create_inliner_pass());
    pm.add_pass(create_canonicalizer_pass());
    pm.add_pass(create_cse_pass());
    pm.add_pass(create_symbol_dce_pass());
    pm.add_pass(create_convert_elementwise_to_linalg_pass());
    // The constructor takes no options; the HMC loops yield fresh buffers each iteration.
    add_pipeline(
        pm.as_operation_pass_manager(),
        "one-shot-bufferize{allow-return-allocs-from-loops=true}",
    );
    pm.add_pass(create_convert_linalg_to_loops_pass());
    pm.add_pass(create_expand_strided_metadata_pass());
    pm.add_pass(create_lower_affine());
    pm.add_pass(create_scf_to_control_flow());
    pm.add_pass(create_to_llvm());
    pm.add_pass(create_reconcile_unrealized_casts());
    if pm.run(&mut module).is_err() {
        panic!("lowering failed:\n{}", module.as_operation());
    }
    ExecutionEngine::new(&module, 2, &[], false, false)
}

#[repr(C)]
struct MemRef2 {
    allocated: *mut f64,
    aligned: *mut f64,
    offset: i64,
    sizes: [i64; 2],
    strides: [i64; 2],
}

const NUM_SAMPLES: usize = 1000;

fn hmc_samples(engine: &ExecutionEngine, seed: (i64, i64), mean: f64, stddev: f64) -> Vec<f64> {
    type Run = unsafe extern "C" fn(i64, i64, f64, f64, *mut MemRef2);
    let run: Run = unsafe { std::mem::transmute(engine.lookup("_mlir_ciface_run")) };
    let mut out = vec![0.0; NUM_SAMPLES];
    let mut memref = MemRef2 {
        allocated: out.as_mut_ptr(),
        aligned: out.as_mut_ptr(),
        offset: 0,
        sizes: [NUM_SAMPLES as i64, 1],
        strides: [1, 1],
    };
    unsafe { run(seed.0, seed.1, mean, stddev, &mut memref) };
    out
}

fn moments(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

// End to end: HMC on N(3, 0.5²) through Impulse expansion, the Impulse derivative rules,
// reverse-mode AD, our Impulse/Enzyme lowerings, and the JIT. No warmup: expand-impulse's
// mass-matrix adaptation currently yields mismatched types from impulse.if.
#[test]
fn hmc_gaussian_samples() {
    let engine = compile(&HMC_RUN.replace("NN", &NUM_SAMPLES.to_string()));

    let samples = hmc_samples(&engine, (42, 7), 3.0, 0.5);
    let (mean, std) = moments(&samples);
    assert!((mean - 3.0).abs() < 0.1, "mean {mean}, expected 3.0");
    assert!((std - 0.5).abs() < 0.1, "std {std}, expected 0.5");

    let other = hmc_samples(&engine, (1, 2), 3.0, 0.5);
    assert_ne!(samples, other, "different seeds produced identical chains");
}

const HMC_FILE: &str = "mlir/impulse/hmc_gaussian.mlir";
const HMC_RUN: &str = include_str!("mlir/impulse/hmc_gaussian.mlir");
