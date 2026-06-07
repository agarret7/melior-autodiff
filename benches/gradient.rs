use criterion::{black_box, criterion_group, criterion_main, Criterion};
use melior::pass::conversion::{
    create_reconcile_unrealized_casts, create_scf_to_control_flow, create_to_llvm,
};
use melior::pass::transform::{create_canonicalizer, create_cse, create_inliner, create_symbol_dce};
use melior::pass::PassManager;
use melior::{dialect::DialectRegistry, ir::Module, utility::register_all_dialects, Context, ExecutionEngine};
use melior_autodiff::{
    enzymeCreateConvertEnzymeToMemRefPass, enzymeCreateDifferentiatePass,
    enzymeRegisterDialectExtensions, enzymeRegisterPasses, enzyme_dialect_handle,
};
use mlir_sys::mlirDialectHandleLoadDialect;
use std::sync::Once;

static PASSES_REGISTERED: Once = Once::new();

fn setup_context() -> Context {
    PASSES_REGISTERED.call_once(|| unsafe { enzymeRegisterPasses() });
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    unsafe { enzymeRegisterDialectExtensions(registry.to_raw()) };
    let ctx = Context::new_with_registry(&registry, false);
    ctx.load_all_available_dialects();
    unsafe {
        let handle = enzyme_dialect_handle();
        mlirDialectHandleLoadDialect(handle, ctx.to_raw());
    }
    ctx
}

fn add_lowering_passes(pm: &PassManager<'_>) {
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateConvertEnzymeToMemRefPass) });
    pm.add_pass(create_inliner());
    pm.add_pass(create_canonicalizer());
    pm.add_pass(create_cse());
    pm.add_pass(create_symbol_dce());
    pm.add_pass(create_scf_to_control_flow());
    pm.add_pass(create_to_llvm());
    pm.add_pass(create_reconcile_unrealized_casts());
}

// x^2 — minimal function, isolates pipeline overhead
const SQUARE_MODULE: &str = r#"
module {
  func.func @square(%x: f64) -> f64 {
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
  func.func @dsquare(%x: f64, %dr: f64) -> f64
      attributes { llvm.emit_c_interface } {
    %r = enzyme.autodiff @square(%x, %dr) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64) -> f64
    return %r : f64
  }
}
"#;

// x^8 via repeated squaring — more AD work
const POLY_MODULE: &str = r#"
module {
  func.func @poly(%x: f64) -> f64 {
    %x2 = arith.mulf %x,  %x  : f64
    %x4 = arith.mulf %x2, %x2 : f64
    %x8 = arith.mulf %x4, %x4 : f64
    return %x8 : f64
  }
  func.func @dpoly(%x: f64, %dr: f64) -> f64
      attributes { llvm.emit_c_interface } {
    %r = enzyme.autodiff @poly(%x, %dr) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64) -> f64
    return %r : f64
  }
}
"#;

// Full pipeline: parse → differentiate → lower → JIT compile.
// Cost Steel pays once at sampler setup.
fn bench_compile(c: &mut Criterion) {
    let ctx = setup_context();
    let mut group = c.benchmark_group("compile");

    group.bench_function("square", |b| {
        b.iter(|| {
            let pm = PassManager::new(&ctx);
            add_lowering_passes(&pm);
            let mut module = Module::parse(&ctx, black_box(SQUARE_MODULE)).unwrap();
            pm.run(&mut module).unwrap();
            let _engine = ExecutionEngine::new(&module, 2, &[], false, false);
        });
    });

    group.bench_function("poly_x8", |b| {
        b.iter(|| {
            let pm = PassManager::new(&ctx);
            add_lowering_passes(&pm);
            let mut module = Module::parse(&ctx, black_box(POLY_MODULE)).unwrap();
            pm.run(&mut module).unwrap();
            let _engine = ExecutionEngine::new(&module, 2, &[], false, false);
        });
    });

    group.finish();
}

// Per-call gradient cost: compile once, call many times.
fn bench_execute(c: &mut Criterion) {
    let ctx = setup_context();
    let mut group = c.benchmark_group("execute");

    // dsquare(x, 1.0) = 2x
    {
        let pm = PassManager::new(&ctx);
        add_lowering_passes(&pm);
        let mut module = Module::parse(&ctx, SQUARE_MODULE).unwrap();
        pm.run(&mut module).unwrap();
        let engine = ExecutionEngine::new(&module, 3, &[], false, false);

        group.bench_function("dsquare", |b| {
            let mut x: f64 = 3.0;
            let mut dr: f64 = 1.0;
            let mut result: f64 = 0.0;
            b.iter(|| unsafe {
                engine
                    .invoke_packed(
                        "dsquare",
                        &mut [
                            &mut x as *mut f64 as *mut (),
                            &mut dr as *mut f64 as *mut (),
                            &mut result as *mut f64 as *mut (),
                        ],
                    )
                    .unwrap();
                black_box(result);
            });
        });
    }

    // dpoly(x, 1.0) = 8x^7
    {
        let pm = PassManager::new(&ctx);
        add_lowering_passes(&pm);
        let mut module = Module::parse(&ctx, POLY_MODULE).unwrap();
        pm.run(&mut module).unwrap();
        let engine = ExecutionEngine::new(&module, 3, &[], false, false);

        group.bench_function("dpoly_x8", |b| {
            let mut x: f64 = 2.0;
            let mut dr: f64 = 1.0;
            let mut result: f64 = 0.0;
            b.iter(|| unsafe {
                engine
                    .invoke_packed(
                        "dpoly",
                        &mut [
                            &mut x as *mut f64 as *mut (),
                            &mut dr as *mut f64 as *mut (),
                            &mut result as *mut f64 as *mut (),
                        ],
                    )
                    .unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

// Box<dyn Fn> wrapper — transmute at construction, vtable dispatch per call.
fn bench_execute_boxed(c: &mut Criterion) {
    let ctx = setup_context();
    let mut group = c.benchmark_group("execute_boxed");

    {
        let pm = PassManager::new(&ctx);
        add_lowering_passes(&pm);
        let mut module = Module::parse(&ctx, SQUARE_MODULE).unwrap();
        pm.run(&mut module).unwrap();
        let engine = ExecutionEngine::new(&module, 3, &[], false, false);

        let dsquare = melior_autodiff::lookup_jit_fn!(&engine, "dsquare", fn(x: f64, dr: f64) -> f64);

        group.bench_function("dsquare", |b| {
            b.iter(|| black_box(dsquare(black_box(3.0), black_box(1.0))));
        });
    }

    {
        let pm = PassManager::new(&ctx);
        add_lowering_passes(&pm);
        let mut module = Module::parse(&ctx, POLY_MODULE).unwrap();
        pm.run(&mut module).unwrap();
        let engine = ExecutionEngine::new(&module, 3, &[], false, false);

        let dpoly = melior_autodiff::lookup_jit_fn!(&engine, "dpoly", fn(x: f64, dr: f64) -> f64);

        group.bench_function("dpoly_x8", |b| {
            b.iter(|| black_box(dpoly(black_box(2.0), black_box(1.0))));
        });
    }

    group.finish();
}

// Raw function pointer vs invoke_packed — same JIT'd code, different call paths.
// invoke_packed goes through a void(**)(void**) trampoline; lookup gives the
// native C ABI pointer directly.
fn bench_execute_raw(c: &mut Criterion) {
    let ctx = setup_context();
    let mut group = c.benchmark_group("execute_raw");

    {
        let pm = PassManager::new(&ctx);
        add_lowering_passes(&pm);
        let mut module = Module::parse(&ctx, SQUARE_MODULE).unwrap();
        pm.run(&mut module).unwrap();
        let engine = ExecutionEngine::new(&module, 3, &[], false, false);

        // invoke_packed baseline
        group.bench_function("dsquare_invoke_packed", |b| {
            let mut x: f64 = 3.0;
            let mut dr: f64 = 1.0;
            let mut result: f64 = 0.0;
            b.iter(|| unsafe {
                engine
                    .invoke_packed(
                        "dsquare",
                        &mut [
                            &mut x as *mut f64 as *mut (),
                            &mut dr as *mut f64 as *mut (),
                            &mut result as *mut f64 as *mut (),
                        ],
                    )
                    .unwrap();
                black_box(result);
            });
        });

        // raw function pointer: lookup once, call directly with C ABI
        let raw = engine.lookup("dsquare");
        assert!(!raw.is_null(), "lookup returned null for dsquare");
        let dsquare_fn: unsafe extern "C" fn(f64, f64) -> f64 =
            unsafe { std::mem::transmute(raw) };

        group.bench_function("dsquare_raw_ptr", |b| {
            b.iter(|| {
                let result = unsafe { dsquare_fn(black_box(3.0), black_box(1.0)) };
                black_box(result);
            });
        });
    }

    {
        let pm = PassManager::new(&ctx);
        add_lowering_passes(&pm);
        let mut module = Module::parse(&ctx, POLY_MODULE).unwrap();
        pm.run(&mut module).unwrap();
        let engine = ExecutionEngine::new(&module, 3, &[], false, false);

        group.bench_function("dpoly_x8_invoke_packed", |b| {
            let mut x: f64 = 2.0;
            let mut dr: f64 = 1.0;
            let mut result: f64 = 0.0;
            b.iter(|| unsafe {
                engine
                    .invoke_packed(
                        "dpoly",
                        &mut [
                            &mut x as *mut f64 as *mut (),
                            &mut dr as *mut f64 as *mut (),
                            &mut result as *mut f64 as *mut (),
                        ],
                    )
                    .unwrap();
                black_box(result);
            });
        });

        let raw = engine.lookup("dpoly");
        assert!(!raw.is_null(), "lookup returned null for dpoly");
        let dpoly_fn: unsafe extern "C" fn(f64, f64) -> f64 =
            unsafe { std::mem::transmute(raw) };

        group.bench_function("dpoly_x8_raw_ptr", |b| {
            b.iter(|| {
                let result = unsafe { dpoly_fn(black_box(2.0), black_box(1.0)) };
                black_box(result);
            });
        });
    }

    group.finish();
}

// Zero vs non-zero seed — shows the strong_zero select guard cost at runtime.
fn bench_strong_zero(c: &mut Criterion) {
    let ctx = setup_context();

    let pm = PassManager::new(&ctx);
    add_lowering_passes(&pm);
    let mut module = Module::parse(&ctx, SQUARE_MODULE).unwrap();
    pm.run(&mut module).unwrap();
    let engine = ExecutionEngine::new(&module, 3, &[], false, false);

    let mut group = c.benchmark_group("strong_zero");

    group.bench_function("nonzero_seed", |b| {
        let mut x: f64 = 3.0;
        let mut dr: f64 = 1.0;
        let mut result: f64 = 0.0;
        b.iter(|| unsafe {
            engine
                .invoke_packed(
                    "dsquare",
                    &mut [
                        &mut x as *mut f64 as *mut (),
                        &mut dr as *mut f64 as *mut (),
                        &mut result as *mut f64 as *mut (),
                    ],
                )
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("zero_seed", |b| {
        let mut x: f64 = 3.0;
        let mut dr: f64 = 0.0;
        let mut result: f64 = 0.0;
        b.iter(|| unsafe {
            engine
                .invoke_packed(
                    "dsquare",
                    &mut [
                        &mut x as *mut f64 as *mut (),
                        &mut dr as *mut f64 as *mut (),
                        &mut result as *mut f64 as *mut (),
                    ],
                )
                .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_compile, bench_execute, bench_execute_boxed, bench_execute_raw, bench_strong_zero);
criterion_main!(benches);
