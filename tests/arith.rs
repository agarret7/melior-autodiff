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
use common::{gen_autodiff, setup_context};

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
    if src.contains("func.func @square") {
        gen_autodiff(ctx, module, "dsquare", "square",
            &["f64", "f64"], &["f64"],
            &[Activity::Active], &[Activity::ActiveNoNeed], true);
    }
    if src.contains("func.func @cube") {
        gen_autodiff(ctx, module, "dcube", "cube",
            &["f64", "f64"], &["f64"],
            &[Activity::Active], &[Activity::ActiveNoNeed], true);
    }
    if src.contains("func.func @poly") {
        gen_autodiff(ctx, module, "dpoly", "poly",
            &["f64", "f64"], &["f64"],
            &[Activity::Active], &[Activity::ActiveNoNeed], true);
    }
    if src.contains("func.func @prod") {
        gen_autodiff(ctx, module, "dprod_dx", "prod",
            &["f64", "f64", "f64", "f64"], &["f64"],
            &[Activity::Active, Activity::Const], &[Activity::ActiveNoNeed], true);
    }
}

fn call2(engine: &ExecutionEngine, name: &str, x: f64, y: f64) -> f64 {
    let mut x = x;
    let mut y = y;
    let mut out = 0.0_f64;
    unsafe {
        engine
            .invoke_packed(name, &mut [
                &mut x as *mut f64 as *mut (),
                &mut y as *mut f64 as *mut (),
                &mut out as *mut f64 as *mut (),
            ])
            .unwrap_or_else(|_| panic!("{name} invoke failed"));
    }
    out
}

fn finite_diff(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-5;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

// f(x) = x²,  f'(x) = 2x  — end-to-end reverse-mode smoke test
#[test]
fn grad_square() {
    let ctx = setup_context();
    let (_, engine) = compile(&ctx, r#"
module {
  func.func @square(%x: f64) -> f64 {
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
}
"#);
    // dsquare(x=3, dr=1) = 2*3 = 6
    let result = call2(&engine, "dsquare", 3.0, 1.0);
    assert!(approx(result, 6.0), "dsquare(3, 1) = {result}, expected 6");
}

// f(x) = x³,  f'(x) = 3x²
#[test]
fn grad_cube() {
    let ctx = setup_context();
    let (_, engine) = compile(&ctx, r#"
module {
  func.func @cube(%x: f64) -> f64 {
    %x2 = arith.mulf %x, %x : f64
    %x3 = arith.mulf %x2, %x : f64
    return %x3 : f64
  }
}
"#);
    for x in [-2.0, -1.0, 0.5, 1.0, 3.0] {
        let enzyme = call2(&engine, "dcube", x, 1.0);
        let fd = finite_diff(|t| t * t * t, x);
        assert!(approx(enzyme, fd), "x={x}: enzyme={enzyme}, fd={fd}");
    }
}

// f(x) = x² - 3x + 1,  f'(x) = 2x - 3
#[test]
fn grad_polynomial() {
    let ctx = setup_context();
    let (_, engine) = compile(&ctx, r#"
module {
  func.func @poly(%x: f64) -> f64 {
    %c3 = arith.constant 3.0 : f64
    %c1 = arith.constant 1.0 : f64
    %x2 = arith.mulf %x, %x : f64
    %t  = arith.mulf %c3, %x : f64
    %y  = arith.subf %x2, %t : f64
    %y1 = arith.addf %y, %c1 : f64
    return %y1 : f64
  }
}
"#);
    for x in [-2.0, 0.0, 1.5, 3.0] {
        let enzyme = call2(&engine, "dpoly", x, 1.0);
        let fd = finite_diff(|t| t * t - 3.0 * t + 1.0, x);
        assert!(approx(enzyme, fd), "x={x}: enzyme={enzyme}, fd={fd}");
    }
}

// f(x, y) = x * y,  ∂f/∂x = y
#[test]
fn grad_product_wrt_x() {
    let ctx = setup_context();
    let (_, engine) = compile(&ctx, r#"
module {
  func.func @prod(%x: f64, %y: f64) -> f64 {
    %r = arith.mulf %x, %y : f64
    return %r : f64
  }
}
"#);
    for (x, y) in [(1.0, 2.0), (-3.0, 4.0), (0.5, 0.5)] {
        let mut xa = x;
        let mut dxa = 1.0_f64;
        let mut ya = y;
        let mut dya = 0.0_f64;
        let mut out = 0.0_f64;
        unsafe {
            engine
                .invoke_packed("dprod_dx", &mut [
                    &mut xa as *mut f64 as *mut (),
                    &mut dxa as *mut f64 as *mut (),
                    &mut ya as *mut f64 as *mut (),
                    &mut dya as *mut f64 as *mut (),
                    &mut out as *mut f64 as *mut (),
                ])
                .expect("dprod_dx invoke failed");
        }
        assert!(approx(out, y), "∂(x*y)/∂x at ({x},{y}): enzyme={out}, expected={y}");
    }
}
