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

fn differentiate(src: &str) -> String {
    let ctx = create_context();
    let mut module = Module::parse(&ctx, src).expect("parse failed");
    let pm = PassManager::new(&ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateExpandImpulsePass) });
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateOutlineEnzymeFromRegionPass) });
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });
    if pm.run(&mut module).is_err() {
        panic!("differentiation failed:\n{}", module.as_operation());
    }
    module.as_operation().to_string()
}

// Reverse-mode wrapper around `@f(%x: $in) -> $out` with an active input and result.
fn reverse(body: &str, input: &str, output: &str) -> String {
    differentiate(&format!(
        r#"
module {{
  func.func @f(%x: {input}) -> {output} {{
{body}
  }}
  func.func @df(%x: {input}, %dr: {output}) -> {input} {{
    %d = enzyme.autodiff @f(%x, %dr) {{
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    }} : ({input}, {output}) -> {input}
    return %d : {input}
  }}
}}
"#
    ))
}

fn diffe_body(ir: &str) -> &str {
    let start = ir.find("@diffef(").expect("no @diffef generated");
    &ir[start..]
}

#[test]
fn reverse_dynamic_slice() {
    let ir = reverse(
        r#"    %i = arith.constant dense<1> : tensor<i64>
    %y = impulse.dynamic_slice %x, %i {slice_sizes = array<i64: 2>}
      : (tensor<4xf64>, tensor<i64>) -> tensor<2xf64>
    return %y : tensor<2xf64>"#,
        "tensor<4xf64>",
        "tensor<2xf64>",
    );
    assert!(diffe_body(&ir).contains("impulse.dynamic_update_slice"), "{ir}");
}

#[test]
fn reverse_dynamic_update_slice() {
    let ir = reverse(
        r#"    %i = arith.constant dense<1> : tensor<i64>
    %u = arith.constant dense<[7.0, 8.0]> : tensor<2xf64>
    %y = impulse.dynamic_update_slice %x, %u, %i
      : (tensor<4xf64>, tensor<2xf64>, tensor<i64>) -> tensor<4xf64>
    return %y : tensor<4xf64>"#,
        "tensor<4xf64>",
        "tensor<4xf64>",
    );
    assert!(diffe_body(&ir).contains("impulse.dynamic_update_slice"), "{ir}");
}

#[test]
fn reverse_slice() {
    let ir = reverse(
        r#"    %y = impulse.slice %x {start_indices = array<i64: 1>, limit_indices = array<i64: 3>,
                           strides = array<i64: 1>} : (tensor<4xf64>) -> tensor<2xf64>
    return %y : tensor<2xf64>"#,
        "tensor<4xf64>",
        "tensor<2xf64>",
    );
    assert!(diffe_body(&ir).contains("impulse.dynamic_update_slice"), "{ir}");
}

#[test]
fn reverse_reshape() {
    let ir = reverse(
        r#"    %y = impulse.reshape %x : (tensor<4xf64>) -> tensor<2x2xf64>
    return %y : tensor<2x2xf64>"#,
        "tensor<4xf64>",
        "tensor<2x2xf64>",
    );
    assert!(diffe_body(&ir).contains("impulse.reshape"), "{ir}");
}

// HMC on a Gaussian: expand-impulse emits enzyme.autodiff_region for ∇logpdf, which must
// outline and differentiate through impulse.{dynamic_slice, dynamic_update_slice, slice, reshape}.
#[test]
fn hmc_gaussian_differentiates() {
    let ir = differentiate(HMC_GAUSSIAN);
    assert!(!ir.contains("enzyme.autodiff_region"), "{ir}");
    assert!(!ir.contains("enzyme.autodiff "), "{ir}");
    assert!(ir.contains("@diffe"), "{ir}");
}

const HMC_GAUSSIAN: &str = r#"
module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %mean : tensor<2xui64>, tensor<f64>
  }
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64> {
    %half = arith.constant dense<-0.5> : tensor<f64>
    %d = arith.subf %x, %mean : tensor<f64>
    %z = arith.divf %d, %stddev : tensor<f64>
    %z2 = arith.mulf %z, %z : tensor<f64>
    %lp = arith.mulf %half, %z2 : tensor<f64>
    return %lp : tensor<f64>
  }
  func.func @model(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }
  func.func @hmc(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<10x1xf64>, tensor<10x2xi1>, tensor<10xf64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:9 = impulse.infer @model(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
        name = "hmc", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 10 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<10x1xf64>, tensor<10x2xi1>, tensor<10xf64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<10x1xf64>, tensor<10x2xi1>, tensor<10xf64>, tensor<2xui64>
  }
}
"#;

/// Appends textual pipeline elements to `pm`. melior's `parse_pass_pipeline` uses
/// `mlirParsePassPipeline`, which replaces the passes already in the manager.
fn add_pipeline(pm: OperationPassManager, elements: &str) {
    unsafe extern "C" fn print_error(message: mlir_sys::MlirStringRef, _: *mut std::ffi::c_void) {
        let bytes = unsafe { std::slice::from_raw_parts(message.data as *const u8, message.length) };
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

const HMC_RUN: &str = r#"
module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %mean : tensor<2xui64>, tensor<f64>
  }
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64> {
    %half = arith.constant dense<-0.5> : tensor<f64>
    %d = arith.subf %x, %mean : tensor<f64>
    %z = arith.divf %d, %stddev : tensor<f64>
    %z2 = arith.mulf %z, %z : tensor<f64>
    %lp = arith.mulf %half, %z2 : tensor<f64>
    return %lp : tensor<f64>
  }
  func.func private @model(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }
  func.func private @hmc(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<NNx1xf64> {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:9 = impulse.infer @model(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
        name = "hmc", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = NN }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<NNx1xf64>, tensor<NNx2xi1>, tensor<NNxf64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0 : tensor<NNx1xf64>
  }
  func.func @run(%seed0: i64, %seed1: i64, %mean: f64, %stddev: f64, %out: memref<NNx1xf64>) attributes { llvm.emit_c_interface } {
    %u0 = builtin.unrealized_conversion_cast %seed0 : i64 to ui64
    %u1 = builtin.unrealized_conversion_cast %seed1 : i64 to ui64
    %rng = tensor.from_elements %u0, %u1 : tensor<2xui64>
    %m = tensor.from_elements %mean : tensor<f64>
    %sd = tensor.from_elements %stddev : tensor<f64>
    %samples = func.call @hmc(%rng, %m, %sd) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> tensor<NNx1xf64>
    bufferization.materialize_in_destination %samples in writable %out : (tensor<NNx1xf64>, memref<NNx1xf64>) -> ()
    return
  }
}
"#;
