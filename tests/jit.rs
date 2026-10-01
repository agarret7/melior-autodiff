use melior::pass::conversion::{
    create_reconcile_unrealized_casts, create_scf_to_control_flow, create_to_llvm,
};
use melior::pass::transform::{
    create_canonicalizer_pass, create_cse_pass, create_inliner_pass, create_symbol_dce_pass,
};
use melior::pass::PassManager;
use melior::{ir::Module, ExecutionEngine};
use melior_autodiff::{
    enzymeCreateConvertEnzymeToMemRefPass, enzymeCreateDifferentiatePass, Activity,
};

mod common;
use common::{gen_autodiff, setup_context};

fn parse_square_with_c_interface(ctx: &melior::Context) -> Module<'_> {
    let module = Module::parse(
        ctx,
        r#"
module {
  func.func @square(%x: f64) -> f64 {
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
}
"#,
    )
    .expect("failed to parse jit test module");
    gen_autodiff(
        ctx,
        &module,
        "dsquare",
        "square",
        &["f64", "f64"],
        &["f64"],
        &[Activity::Active],
        &[Activity::ActiveNoNeed],
        true,
    );
    module
}

#[test]
fn jit_gradient_of_square() {
    let ctx = setup_context();
    let mut module = parse_square_with_c_interface(&ctx);

    // Differentiate, inline, and clean up.
    let pm = PassManager::new(&ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateConvertEnzymeToMemRefPass) });
    pm.add_pass(create_inliner_pass());
    pm.add_pass(create_canonicalizer_pass());
    pm.add_pass(create_cse_pass());
    pm.add_pass(create_symbol_dce_pass());

    // Lower to LLVM.
    pm.add_pass(create_scf_to_control_flow());
    pm.add_pass(create_to_llvm());
    pm.add_pass(create_reconcile_unrealized_casts());

    eprintln!("before lowering:\n{}", module.as_operation());
    pm.run(&mut module).expect("lowering pipeline failed");
    eprintln!("after lowering:\n{}", module.as_operation());

    let engine = ExecutionEngine::new(&module, 2, &[], false, false);

    // dsquare(x, dr) = dr * 2*x  →  dsquare(3.0, 1.0) = 6.0
    let mut x: f64 = 3.0;
    let mut dr: f64 = 1.0;
    let mut result: f64 = 0.0;
    unsafe {
        engine
            .invoke_packed(
                "dsquare",
                &mut [
                    &mut x as *mut f64 as *mut (),
                    &mut dr as *mut f64 as *mut (),
                    &mut result as *mut f64 as *mut (),
                ],
            )
            .expect("invoke_packed failed");
    }

    assert!(
        (result - 6.0).abs() < 1e-10,
        "dsquare(3, 1) should be 6, got {result}"
    );
}
