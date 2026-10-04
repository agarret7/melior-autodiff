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

fn call(engine: &ExecutionEngine, name: &str, args: &[f64]) -> f64 {
    let mut args = args.to_vec();
    let mut out = 0.0_f64;
    let mut ptrs = args
        .iter_mut()
        .map(|a| a as *mut f64 as *mut ())
        .collect::<Vec<_>>();
    ptrs.push(&mut out as *mut f64 as *mut ());
    unsafe { engine.invoke_packed(name, &mut ptrs) }
        .unwrap_or_else(|_| panic!("{name} invoke failed"));
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
    let engine = compile(
        r#"
module {
  func.func @square(%x: f64) -> f64 {
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
  func.func @dsquare(%x: f64, %dr: f64) -> f64 attributes { llvm.emit_c_interface } {
    %d = enzyme.autodiff @square(%x, %dr) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64) -> f64
    return %d : f64
  }
}
"#,
    );
    let result = call(&engine, "dsquare", &[3.0, 1.0]);
    assert!(approx(result, 6.0), "dsquare(3, 1) = {result}, expected 6");
}

// f(x) = x³,  f'(x) = 3x²
#[test]
fn grad_cube() {
    let engine = compile(
        r#"
module {
  func.func @cube(%x: f64) -> f64 {
    %x2 = arith.mulf %x, %x : f64
    %x3 = arith.mulf %x2, %x : f64
    return %x3 : f64
  }
  func.func @dcube(%x: f64, %dr: f64) -> f64 attributes { llvm.emit_c_interface } {
    %d = enzyme.autodiff @cube(%x, %dr) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64) -> f64
    return %d : f64
  }
}
"#,
    );
    for x in [-2.0, -1.0, 0.5, 1.0, 3.0] {
        let enzyme = call(&engine, "dcube", &[x, 1.0]);
        let fd = finite_diff(|t| t * t * t, x);
        assert!(approx(enzyme, fd), "x={x}: enzyme={enzyme}, fd={fd}");
    }
}

// f(x) = x² - 3x + 1,  f'(x) = 2x - 3
#[test]
fn grad_polynomial() {
    let engine = compile(
        r#"
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
  func.func @dpoly(%x: f64, %dr: f64) -> f64 attributes { llvm.emit_c_interface } {
    %d = enzyme.autodiff @poly(%x, %dr) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64) -> f64
    return %d : f64
  }
}
"#,
    );
    for x in [-2.0, 0.0, 1.5, 3.0] {
        let enzyme = call(&engine, "dpoly", &[x, 1.0]);
        let fd = finite_diff(|t| t * t - 3.0 * t + 1.0, x);
        assert!(approx(enzyme, fd), "x={x}: enzyme={enzyme}, fd={fd}");
    }
}

// f(x, y) = x * y,  ∂f/∂x = y.  y is enzyme_const, so it takes no shadow operand.
#[test]
fn grad_product_wrt_x() {
    let engine = compile(
        r#"
module {
  func.func @prod(%x: f64, %y: f64) -> f64 {
    %r = arith.mulf %x, %y : f64
    return %r : f64
  }
  func.func @dprod_dx(%x: f64, %y: f64, %dr: f64) -> f64 attributes { llvm.emit_c_interface } {
    %d = enzyme.autodiff @prod(%x, %y, %dr) {
      activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64, f64) -> f64
    return %d : f64
  }
}
"#,
    );
    for (x, y, dr) in [(1.0, 2.0, 1.0), (-3.0, 4.0, 1.0), (0.5, 0.5, 2.0)] {
        let out = call(&engine, "dprod_dx", &[x, y, dr]);
        assert!(
            approx(out, y * dr),
            "∂(x*y)/∂x at ({x},{y}) seed {dr}: enzyme={out}, expected={}",
            y * dr
        );
    }
}
