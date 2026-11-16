use melior::ir::Module;
use melior::pass::transform::create_canonicalizer_pass;
use melior::pass::PassManager;
use melior::Error;
use melior_autodiff::pass::{self, DerivativeMode};
use melior_autodiff::{utility::create_context, Activity};

mod filecheck;
use filecheck::filecheck;

const SQUARE: &str = "mlir/wrap/square.mlir";

// Runs enzyme-wrap on @square, then Enzyme's usual cleanup, and returns the module's text.
fn wrap(outfn: &str, activity: Activity, mode: DerivativeMode) -> String {
    let context = create_context();
    let src = std::fs::read_to_string(format!("{}/tests/{SQUARE}", env!("CARGO_MANIFEST_DIR")))
        .unwrap();
    let mut module = Module::parse(&context, &src).unwrap();
    let pm = PassManager::new(&context);
    pass::add_differentiate_wrapper_pass(&pm, "square", Some(outfn), &[activity], &[activity], mode)
        .unwrap();
    pm.add_pass(create_canonicalizer_pass());
    pm.add_pass(pass::create_remove_unused_enzyme_ops_pass());
    pm.add_pass(pass::create_mathematic_simplification_pass());
    pm.run(&mut module).expect("enzyme-wrap pipeline failed");
    module.as_operation().to_string()
}

#[test]
fn reverse_mode() {
    let text = wrap("dsquare", Activity::Active, DerivativeMode::ReverseModeCombined);
    filecheck(&text, SQUARE, Some("REVERSE"));
}

#[test]
fn forward_mode() {
    let text = wrap("fwd_square", Activity::Dup, DerivativeMode::ForwardMode);
    filecheck(&text, SQUARE, Some("FORWARD"));
}

// enzyme-wrap asserts on activities it can't parse, so they're rejected before it runs.
#[test]
fn rejects_unsupported_activity() {
    let context = create_context();
    let pm = PassManager::new(&context);
    for activity in [Activity::ActiveNoNeed, Activity::ConstNoNeed] {
        let result = pass::add_differentiate_wrapper_pass(
            &pm,
            "square",
            Some("dsquare"),
            &[activity],
            &[Activity::Active],
            DerivativeMode::ReverseModeCombined,
        );
        assert!(matches!(result, Err(Error::ParsePassPipeline(_))), "{activity:?}: {result:?}");
    }
}

// A name that would break the pipeline syntax is rejected rather than spliced into it.
#[test]
fn rejects_invalid_function_name() {
    let context = create_context();
    let pm = PassManager::new(&context);
    for name in ["", "sq uare", "square}", "a,b"] {
        let result = pass::add_differentiate_wrapper_pass(
            &pm,
            name,
            Some("dsquare"),
            &[Activity::Active],
            &[Activity::Active],
            DerivativeMode::ReverseModeCombined,
        );
        assert!(matches!(result, Err(Error::ParsePassPipeline(_))), "{name:?}: {result:?}");
    }
}

// With no `outfn`, enzyme-wrap replaces @square with its derivative under the same name, and
// makes it public, so it can be looked up after JIT compilation.
#[test]
fn replace_mode_is_public() {
    let context = create_context();
    let src = std::fs::read_to_string(format!("{}/tests/{SQUARE}", env!("CARGO_MANIFEST_DIR")))
        .unwrap();
    let mut module = Module::parse(&context, &src).unwrap();
    let pm = PassManager::new(&context);
    pass::add_differentiate_wrapper_pass(
        &pm,
        "square",
        None,
        &[Activity::Active],
        &[Activity::Active],
        DerivativeMode::ReverseModeCombined,
    )
    .unwrap();
    pm.run(&mut module).unwrap();
    let text = module.as_operation().to_string();
    let square = text.lines().find(|line| line.contains("@square(")).unwrap();
    assert!(square.contains("func.func @square(%arg0: f64, %arg1: f64) -> f64"), "{square}");
}
