// Every Enzyme pass constructor builds a pass that runs.

use melior::{
    ir::{
        Module,
        operation::{OperationLike, WalkOrder, WalkResult},
    },
    pass::{Pass, PassManager},
};
use melior_autodiff::{pass, utility::create_context};

type CreatePass = fn() -> Pass;

#[test]
fn every_pass_runs() {
    let constructors: &[(&str, CreatePass)] = &[
        ("enzyme", pass::create_differentiate_pass),
        ("enzyme-batch", pass::create_batch_pass),
        ("enzyme-diff-batch", pass::create_batch_diff_pass),
        ("enzyme-activity-opt", pass::create_enzyme_activity_opt_pass),
        (
            "enzyme-simplify-math",
            pass::create_mathematic_simplification_pass,
        ),
        (
            "remove-unnecessary-enzyme-ops",
            pass::create_remove_unused_enzyme_ops_pass,
        ),
        (
            "flatten-enzyme-caches",
            pass::create_flatten_enzyme_caches_pass,
        ),
        (
            "simplify-memref-cache",
            pass::create_simplify_memref_cache_pass,
        ),
        (
            "enzyme-batch-to-tensor",
            pass::create_enzyme_batch_to_tensor_pass,
        ),
        (
            "convert-enzyme-to-memref",
            pass::create_enzyme_ops_to_mem_ref_pass,
        ),
        ("add-to-op-to-split", pass::create_add_to_op_to_split_pass),
        ("split-multi-results", pass::create_split_multi_results_pass),
        (
            "inline-enzyme-regions",
            pass::create_inline_enzyme_into_region_pass,
        ),
        (
            "outline-enzyme-regions",
            pass::create_outline_enzyme_from_region_pass,
        ),
        (
            "hoist-enzyme-regions",
            pass::create_hoist_enzyme_from_region_pass,
        ),
        (
            "sink-checkpoint-views",
            pass::create_sink_checkpoint_views_pass,
        ),
        (
            "hoist-loop-allocations",
            pass::create_hoist_loop_allocations_pass,
        ),
        (
            "lower-affine-atomic-rmw",
            pass::create_lower_affine_atomic_rmw_pass,
        ),
        ("raise-llvm-ext", pass::create_raise_llvm_ext_pass),
        ("lower-llvm-ext", pass::create_lower_llvm_ext_pass),
        (
            "lower-enzyme-binomial-progress",
            pass::create_lower_binomial_progress_pass,
        ),
        (
            "print-activity-analysis",
            pass::create_print_activity_analysis_pass,
        ),
    ];
    let ctx = create_context();
    for (name, create) in constructors {
        let pm = PassManager::new(&ctx);
        pm.add_pass(create());
        let mut module = Module::parse(&ctx, "module {}").unwrap();
        pm.run(&mut module)
            .unwrap_or_else(|e| panic!("{name} failed: {e}"));
    }
}

#[test]
fn enzyme_lowering_runs() {
    let ctx = create_context();
    let pm = PassManager::new(&ctx);
    pass::add_enzyme_lowering(&pm);
    let mut module = Module::parse(&ctx, "module {}").unwrap();
    pm.run(&mut module).unwrap();
}

// Every pass constructor Enzyme generates is wrapped in `pass` or skipped here with a reason, so a
// pass Enzyme adds upstream fails this test until it's one or the other.
#[test]
fn every_generated_pass_is_wrapped() {
    let skipped = [
        // Added as text, with its options, by `add_differentiate_wrapper_pass`.
        "mlirCreateEnzymeDifferentiateWrapperPass",
        // Deprecated in favor of `add-to-op-to-split`.
        "mlirCreateEnzymeAddToOpToIndexAndLoadPass",
        // Enzyme's test-only passes.
        "mlirCreateEnzymePrintAliasAnalysisPass",
        "mlirCreateEnzymePrintBaseObjectPass",
        // Lowers Enzyme's Impulse dialect, which this crate doesn't wrap yet.
        "mlirCreateEnzymeExpandImpulsePass",
    ];
    let generated: Vec<&str> = env!("ENZYME_PASS_CONSTRUCTORS")
        .split(',')
        .filter(|name| !name.is_empty())
        .collect();
    let wrapped: Vec<&str> = include_str!("../src/pass.rs")
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|word| word.starts_with("mlirCreateEnzyme"))
        .collect();

    let missing: Vec<_> = generated
        .iter()
        .filter(|name| !wrapped.contains(name) && !skipped.contains(name))
        .collect();
    assert!(missing.is_empty(), "not wrapped in `pass`: {missing:?}");
    let stale: Vec<_> = skipped
        .iter()
        .filter(|name| !generated.contains(name))
        .collect();
    assert!(
        stale.is_empty(),
        "skipped, but Enzyme no longer generates: {stale:?}"
    );
}

// With `annotate`, the analysis is kept on the IR. In f(x) = x + 2 * 2, the constant and the
// product of constants are inactive, and the sum with x is active.
#[test]
fn activity_analysis_annotates() {
    melior_autodiff::utility::register_passes();
    let ctx = create_context();
    let pm = PassManager::new(&ctx);
    melior::utility::parse_pass_pipeline(
        pm.as_operation_pass_manager(),
        "builtin.module(print-activity-analysis{annotate=true})",
    )
    .unwrap();
    let mut module = Module::parse(
        &ctx,
        r#"
        func.func @f(%x: f64) -> f64 {
          %c = arith.constant 2.0 : f64
          %c2 = arith.mulf %c, %c : f64
          %y = arith.addf %x, %c2 : f64
          return %y : f64
        }
        "#,
    )
    .unwrap();
    pm.run(&mut module).unwrap();

    let mut activity = Vec::new();
    module.as_operation().walk(WalkOrder::PreOrder, |op| {
        let name = op.name().as_string_ref().as_str().unwrap().to_owned();
        if name.starts_with("arith.") {
            let constant = op.attribute("enzyme.ici").unwrap().to_string();
            activity.push((name, constant));
        }
        WalkResult::Advance
    });
    let expected = [
        ("arith.constant", "true"),
        ("arith.mulf", "true"),
        ("arith.addf", "false"),
    ];
    let activity: Vec<_> = activity
        .iter()
        .map(|(name, constant)| (name.as_str(), constant.as_str()))
        .collect();
    assert_eq!(activity, expected, "(op, is constant)");
}

// `register_passes` makes every pass nameable in a textual pipeline.
#[test]
fn passes_are_registered_by_name() {
    melior_autodiff::utility::register_passes();
    let ctx = create_context();
    let pm = PassManager::new(&ctx);
    melior::utility::parse_pass_pipeline(
        pm.as_operation_pass_manager(),
        "builtin.module(enzyme, flatten-enzyme-caches, convert-enzyme-to-memref)",
    )
    .unwrap();
    let mut module = Module::parse(&ctx, "module {}").unwrap();
    pm.run(&mut module).unwrap();
}
