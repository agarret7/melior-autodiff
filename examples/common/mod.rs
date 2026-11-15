//! What the examples share: one simple way to get from differentiated MLIR to code melior's
//! `ExecutionEngine` can run. It's a recipe to copy and adapt, not part of the crate's API; real
//! pipelines will want their own bufferization, fusion, parallel loops or target.
#![allow(dead_code)]

use melior::{
    dialect::func,
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        r#type::FunctionType,
        Block, BlockLike, Location, Module, Region, RegionLike, Type,
    },
    pass::PassManager,
    utility::{parse_pass_pipeline, register_all_passes},
    Context, ExecutionEngine,
};

/// Lowers `module`, already free of Enzyme ops (see `pass::add_enzyme_lowering`), to the LLVM
/// dialect and JIT-compiles it.
///
/// The last step uses MLIR's *bare-pointer* calling convention: a statically shaped `memref`
/// argument becomes a plain pointer to its data, and scalars are passed by value. So
/// `func.func @f(%x: memref<4xf64>, %dx: memref<4xf64>, %y: f64)` is called from Rust as
/// `extern "C" fn(*const f64, *mut f64, f64)`, just like Enzyme's C `__enzyme_autodiff`.
/// Dynamically shaped memrefs need MLIR's full descriptors instead.
pub fn jit(context: &Context, module: &mut Module) -> ExecutionEngine {
    register_all_passes();
    let pm = PassManager::new(context);
    parse_pass_pipeline(
        pm.as_operation_pass_manager(),
        "builtin.module(\
            inline, canonicalize, cse, symbol-dce, \
            convert-elementwise-to-linalg, one-shot-bufferize, convert-linalg-to-loops, \
            convert-scf-to-cf, expand-strided-metadata, lower-affine, \
            convert-math-to-llvm, convert-arith-to-llvm, finalize-memref-to-llvm, \
            convert-cf-to-llvm, convert-func-to-llvm{use-bare-ptr-memref-call-conv=1}, \
            reconcile-unrealized-casts)",
    )
    .expect("invalid pipeline");
    if let Err(error) = pm.run(module) {
        panic!("lowering failed: {error}\n{}", module.as_operation());
    }
    ExecutionEngine::new(module, 2, &[], false, false).expect("JIT compilation failed")
}

/// Appends a public `func.func @name(inputs...)` with no results to `module`. `body` fills its
/// block, whose arguments are the inputs; the closing `func.return` is added after it.
pub fn add_function<'c>(
    context: &'c Context,
    module: &Module<'c>,
    name: &str,
    inputs: &[Type<'c>],
    body: impl FnOnce(&Block<'c>),
) {
    let location = Location::unknown(context);
    let block = Block::new(inputs, &vec![location; inputs.len()]).unwrap();
    body(&block);
    block.append_operation(func::r#return(&[], location));
    let region = Region::new();
    region.append_block(block);
    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, name),
        TypeAttribute::new(FunctionType::new(context, inputs, &[]).into()),
        region,
        &[],
        location,
    ));
}

/// Looks up the JIT-compiled function `name`.
///
/// # Safety
///
/// `F` must be an `extern "C" fn` type matching the function's signature under the bare-pointer
/// convention.
pub unsafe fn lookup<F: Copy>(engine: &ExecutionEngine, name: &str) -> F {
    let pointer = engine.lookup(name);
    assert!(!pointer.is_null(), "no function @{name}");
    assert_eq!(size_of::<F>(), size_of::<*mut ()>());
    unsafe { std::mem::transmute_copy(&pointer) }
}

/// The central finite difference of `f` at `x`, to check derivatives against.
pub fn finite_difference(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-6;
    (f(x + h) - f(x - h)) / (2.0 * h)
}
